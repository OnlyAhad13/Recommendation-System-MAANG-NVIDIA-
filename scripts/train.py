#!/usr/bin/env python3
"""
Training entrypoint script for the recommendation system.
"""

import argparse
import logging
import warnings
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from src.config import ModelConfig
from src.trainer import ProductionTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Mixed precision - only if GPU available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"Found {len(gpus)} GPU(s). Enabling mixed precision.")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
else:
    logger.info("No GPU found. Using float32.")


def main():
    parser = argparse.ArgumentParser(
        description='Train Enterprise Recommendation System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to preprocessed data pickle file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./outputs/models/experiment_001',
        help='Output directory for models and logs'
    )
    
    # Model architecture
    parser.add_argument(
        '--embedding_dim', 
        type=int, 
        default=64,
        help='Embedding dimension'
    )
    parser.add_argument(
        '--cross_layers', 
        type=int, 
        default=1,
        help='Number of cross layers in DCN'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=2048,
        help='Training batch size'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=5,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.001,
        help='Learning rate for retrieval task'
    )
    
    # Negative sampling
    parser.add_argument(
        '--negative_sampling', 
        choices=['random', 'hard', 'mixed'], 
        default='mixed',
        help='Negative sampling strategy'
    )
    parser.add_argument(
        '--num_hard_negatives', 
        type=int, 
        default=20,
        help='Number of hard negatives'
    )
    parser.add_argument(
        '--num_random_negatives', 
        type=int, 
        default=30,
        help='Number of random negatives'
    )
    
    # Multi-task weights
    parser.add_argument(
        '--ctr_weight', 
        type=float, 
        default=0.2,
        help='Weight for CTR loss'
    )
    parser.add_argument(
        '--rating_weight', 
        type=float, 
        default=0.2,
        help='Weight for rating loss'
    )
    
    # System settings
    parser.add_argument(
        '--distributed_strategy', 
        choices=['none', 'mirrored', 'multi_worker'], 
        default='none',
        help='Distributed training strategy'
    )
    parser.add_argument(
        '--use_wandb', 
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ModelConfig(
        embedding_dim=args.embedding_dim,
        cross_layers=args.cross_layers,
        batch_size=args.batch_size,
        epochs_retrieval=args.epochs,
        learning_rate_retrieval=args.learning_rate,
        negative_sampling_strategy=args.negative_sampling,
        num_hard_negatives=args.num_hard_negatives,
        num_random_negatives=args.num_random_negatives,
        ctr_weight=args.ctr_weight,
        rating_weight=args.rating_weight,
        distributed_strategy=args.distributed_strategy
    )
    
    # Log configuration
    logger.info("="*80)
    logger.info("TRAINING CONFIGURATION:")
    for k, v in config.to_dict().items():
        logger.info(f"  {k}: {v}")
    logger.info("="*80)
    
    # Create trainer
    trainer = ProductionTrainer(config, args.output_dir)
    
    # Train
    try:
        model, history = trainer.train(args.data)
        logger.info(f"\n✓ Training completed successfully!")
        logger.info(f"  Artifacts saved to: {args.output_dir}")
    except KeyboardInterrupt:
        logger.warning("\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup W&B if used
        if args.use_wandb:
            try:
                import wandb
                if wandb.run:
                    wandb.finish()
            except:
                pass


if __name__ == "__main__":
    main()

