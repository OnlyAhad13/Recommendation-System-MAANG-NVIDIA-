"""
Training orchestration for the recommendation system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .config import ModelConfig
from .models import MultiTaskModel
from .data_processing import DataProcessor, NegativeSampler
from .evaluation import AdvancedMetrics, CustomMetricsCallback

# Optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProductionTrainer:
    """Production-grade trainer for recommendation system."""
    
    def __init__(self, config: ModelConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Distributed strategy
        self.strategy = tf.distribute.get_strategy()
        gpus = tf.config.list_physical_devices('GPU')
        if config.distributed_strategy == "mirrored" and len(gpus) > 1:
            self.strategy = tf.distribute.MirroredStrategy()
        
        # Components
        self.data_processor = DataProcessor(config)
        self.negative_sampler = NegativeSampler(
            config.negative_sampling_strategy, 
            config.num_hard_negatives, 
            config.num_random_negatives
        )
        self.metrics_calculator = AdvancedMetrics()
        
        # Initialize W&B if available
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="enterprise-recsys", 
                    config=asdict(config), 
                    dir=str(self.output_dir)
                )
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
    
    def prepare_datasets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare TensorFlow datasets from raw data."""
        logger.info("Preparing datasets...")
        
        # Feature engineering
        train_df = self.data_processor.engineer_features(
            data['train_df'], data['user_features'], data['item_features'], 'train'
        )
        val_df = self.data_processor.engineer_features(
            data['val_df'], data['user_features'], data['item_features'], 'val'
        )
        
        # Fit negative sampler
        self.negative_sampler.fit(train_df)
        
        # Build vocabularies
        user_vocab = sorted(train_df['user_id'].unique().tolist())
        item_vocab = sorted(train_df['movie_id'].unique().tolist())
        
        # Feature specifications
        feature_specs = {}
        for col in train_df.columns:
            if col in ['user_id', 'movie_id', 'rating', 'y_implicit', 'timestamp']:
                continue
            if pd.api.types.is_numeric_dtype(train_df[col]):
                feature_specs[col] = {'type': 'numerical'}
            else:
                nunique = train_df[col].nunique()
                if nunique < 10000:
                    feature_specs[col] = {'type': 'categorical', 'vocab_size': nunique}
        
        # Create TensorFlow datasets
        def make_ds(df, training=False):
            if df.empty:
                return None
            
            feature_dict = {
                'user_id': df['user_id'].astype(str).values,
                'movie_id': df['movie_id'].astype(str).values
            }
            
            if 'rating' in df.columns:
                feature_dict['rating'] = df['rating'].astype(np.float32).values
            
            if 'y_implicit' in df.columns:
                feature_dict['y_implicit'] = df['y_implicit'].astype(np.float32).values
            elif 'rating' in df.columns:
                feature_dict['y_implicit'] = (df['rating'] >= 3.0).astype(np.float32).values
            
            ds = tf.data.Dataset.from_tensor_slices(feature_dict)
            if training:
                ds = ds.shuffle(50000)
            return ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return {
            'train_ds': make_ds(train_df, True),
            'val_ds': make_ds(val_df),
            'train_df': train_df,
            'val_df': val_df,
            'user_vocab': user_vocab,
            'item_vocab': item_vocab,
            'feature_specs': feature_specs
        }
    
    def train(self, pickle_path: str):
        """Train the recommendation model."""
        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        
        # Load and prepare data
        data = self.data_processor.load_and_validate_data(pickle_path)
        datasets = self.prepare_datasets(data)
        
        # Build model
        with self.strategy.scope():
            model = MultiTaskModel(
                self.config, 
                datasets['user_vocab'], 
                datasets['item_vocab'], 
                datasets['feature_specs']
            )
            
            # Learning rate schedule
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                self.config.learning_rate_retrieval, 
                decay_steps=1000, 
                decay_rate=0.96, 
                staircase=True
            )
            model.compile(optimizer=keras.optimizers.Adagrad(lr))
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                'val_root_mean_squared_error', 
                patience=5, 
                restore_best_weights=True, 
                verbose=1,
                mode='min'
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.output_dir / 'best_model.keras'), 
                'val_loss', 
                save_best_only=True, 
                verbose=1
            ),
            keras.callbacks.CSVLogger(str(self.output_dir / 'training_log.csv')),
            keras.callbacks.TensorBoard(str(self.output_dir / 'logs')),
            CustomMetricsCallback(self.output_dir)
        ]
        
        if WANDB_AVAILABLE and wandb.run:
            callbacks.append(
                wandb.keras.WandbCallback(
                    save_model=False, 
                    log_weights=False, 
                    log_gradients=False
                )
            )
        
        # Train model
        logger.info(f"Training for {self.config.epochs_retrieval} epochs...")
        history = model.fit(
            datasets['train_ds'], 
            validation_data=datasets['val_ds'],
            epochs=self.config.epochs_retrieval, 
            callbacks=callbacks, 
            verbose=1
        )
        
        # Post-training steps
        self._evaluate(model, datasets)
        self._save_artifacts(model, datasets)
        
        if FAISS_AVAILABLE:
            self._build_faiss(model, datasets['item_vocab'])
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        
        return model, history
    
    def _evaluate(self, model, datasets):
        """Evaluate model performance."""
        logger.info("Evaluating model...")
        
        if datasets['val_ds'] is None or datasets['val_df'].empty:
            logger.warning("No validation data available")
            return
        
        # Get item embeddings
        item_embs = self._get_item_embeddings(model, datasets['item_vocab'])
        
        # Sample validation users
        val_sample = datasets['val_df'].sample(
            n=min(1000, len(datasets['val_df'])), 
            random_state=42
        )
        
        # Get user embeddings
        user_embs = model.encoder(
            {'user_id': tf.constant(val_sample['user_id'].values)}, 
            training=False
        )['user_embedding'].numpy()
        
        # Compute similarities
        sims = np.dot(user_embs, item_embs.T)
        
        # Compute metrics
        metrics = {}
        for k in self.config.eval_topk:
            recalls = []
            for sim, true_item in zip(sims, val_sample['movie_id'].values):
                top_k_idx = np.argpartition(-sim, min(k, len(sim)-1))[:k]
                top_k = [
                    datasets['item_vocab'][i] 
                    for i in top_k_idx 
                    if i < len(datasets['item_vocab'])
                ]
                recalls.append(1.0 if true_item in top_k else 0.0)
            metrics[f'recall@{k}'] = np.mean(recalls)
        
        # Log metrics
        logger.info("Evaluation Results:")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")
        
        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        if WANDB_AVAILABLE and wandb.run:
            wandb.log(metrics)
    
    def _get_item_embeddings(self, model, item_vocab):
        """Get embeddings for all items."""
        embs = []
        batch_size = 512
        for i in range(0, len(item_vocab), batch_size):
            batch = item_vocab[i:i+batch_size]
            emb = model.encoder(
                {'movie_id': tf.constant(batch)}, 
                training=False
            )['item_embedding'].numpy()
            embs.append(emb)
        return np.vstack(embs)
    
    def _save_artifacts(self, model, datasets):
        """Save model and related artifacts."""
        logger.info("Saving artifacts...")
        
        # Save full model
        try:
            model.save(str(self.output_dir / 'model'), save_format='tf')
        except Exception as e:
            logger.warning(f"Failed to save full model: {e}")
        
        # Save encoder
        try:
            model.encoder.save(str(self.output_dir / 'encoder.keras'))
        except Exception as e:
            logger.warning(f"Failed to save encoder: {e}")
        
        # Save vocabularies
        with open(self.output_dir / 'vocabs.json', 'w') as f:
            json.dump({
                'users': datasets['user_vocab'], 
                'items': datasets['item_vocab']
            }, f)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def _build_faiss(self, model, item_vocab):
        """Build FAISS index for fast retrieval."""
        logger.info("Building FAISS index...")
        
        try:
            # Get item embeddings
            embs = self._get_item_embeddings(model, item_vocab).astype(np.float32)
            faiss.normalize_L2(embs)
            
            dim = embs.shape[1]
            n = len(item_vocab)
            
            # Choose index type based on size
            if n > 10000:
                nlist = min(4096, n // 10)
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(embs)
                index.add(embs)
                index.nprobe = min(128, nlist // 4)
            else:
                index = faiss.IndexFlatIP(dim)
                index.add(embs)
            
            # Save index
            faiss.write_index(index, str(self.output_dir / 'faiss.idx'))
            
            # Save item mapping
            with open(self.output_dir / 'item_map.json', 'w') as f:
                json.dump({str(i): item for i, item in enumerate(item_vocab)}, f)
            
            logger.info(f"FAISS index built: {n} items, type={type(index).__name__}")
        except Exception as e:
            logger.error(f"FAISS build failed: {e}")

