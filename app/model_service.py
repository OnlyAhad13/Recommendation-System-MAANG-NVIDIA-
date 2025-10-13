"""
Model service for loading and serving the recommendation model.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import tensorflow as tf
import sys

# Add src to path and import models to register custom classes
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import MultiTaskModel, MultiTowerModel, DeepCrossNetwork

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for loading model and generating recommendations."""
    
    def __init__(self, model_dir: str = "outputs/models/my_model"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.user_vocab = None
        self.item_vocab = None
        self.item_embeddings = None
        self.config = None
        self.version = "1.0.0"
        
    def load_model(self):
        """Load the trained model and vocabularies."""
        logger.info(f"Loading model from {self.model_dir}")
        
        try:
            # Load model
            model_path = self.model_dir / "best_model.keras"
            if model_path.exists():
                self.model = tf.keras.models.load_model(str(model_path), compile=False)
                logger.info(" Model loaded")
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Load vocabularies
            vocab_path = self.model_dir / "vocabs.json"
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    vocabs = json.load(f)
                    self.user_vocab = vocabs['users']
                    self.item_vocab = vocabs['items']
                logger.info(f"Vocabularies loaded: {len(self.user_vocab)} users, {len(self.item_vocab)} items")
            else:
                raise FileNotFoundError(f"Vocabularies not found at {vocab_path}")
            
            # Load config
            config_path = self.model_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(" Config loaded")
            
            # Pre-compute all item embeddings for fast retrieval
            self._precompute_item_embeddings()
            
            logger.info(" Model service ready")
            
        except Exception as e:
            logger.error(f" Failed to load model: {e}")
            raise
    
    def _precompute_item_embeddings(self):
        """Pre-compute embeddings for all items."""
        logger.info("Pre-computing item embeddings...")
        
        all_item_embs = []
        batch_size = 512
        
        for i in range(0, len(self.item_vocab), batch_size):
            batch = self.item_vocab[i:i+batch_size]
            try:
                emb = self.model({'movie_id': tf.constant(batch)}, training=False)['item_embedding']
                all_item_embs.append(emb.numpy())
            except:
                # Fallback if model expects both user_id and movie_id
                dummy_users = [self.user_vocab[0]] * len(batch)
                emb = self.model({'user_id': tf.constant(dummy_users), 'movie_id': tf.constant(batch)}, training=False)['item_embedding']
                all_item_embs.append(emb.numpy())
        
        self.item_embeddings = np.vstack(all_item_embs)
        logger.info(f" Item embeddings computed: {self.item_embeddings.shape}")
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return (self.model is not None and 
                self.user_vocab is not None and 
                self.item_vocab is not None and
                self.item_embeddings is not None)
    
    def get_version(self) -> str:
        """Get model version."""
        return self.version
    
    def recommend(self, user_id: str, k: int = 10, exclude_seen: bool = True) -> List[Dict]:
        """
        Generate top-K recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations
            exclude_seen: Whether to exclude items user has seen
            
        Returns:
            List of recommended items with scores
        """
        # Check if user exists
        if user_id not in self.user_vocab:
            logger.warning(f"User {user_id} not in vocabulary, using cold-start strategy")
            # Cold-start: return popular items
            return self._get_popular_items(k)
        
        # Get user embedding
        try:
            user_emb = self.model({'user_id': tf.constant([user_id])}, training=False)['user_embedding']
            user_emb = user_emb.numpy()[0]
        except:
            # Fallback
            user_emb = self.model(
                {'user_id': tf.constant([user_id]), 'movie_id': tf.constant([self.item_vocab[0]])}, 
                training=False
            )['user_embedding'].numpy()[0]
        
        # Compute similarities with all items
        similarities = np.dot(self.item_embeddings, user_emb)
        
        # Get top-K indices
        top_k_idx = np.argpartition(-similarities, min(k, len(similarities)-1))[:k]
        top_k_idx = top_k_idx[np.argsort(-similarities[top_k_idx])]
        
        # Format results
        recommendations = []
        for rank, idx in enumerate(top_k_idx, 1):
            if idx < len(self.item_vocab):
                recommendations.append({
                    "item_id": self.item_vocab[idx],
                    "score": float(similarities[idx]),
                    "rank": rank
                })
        
        return recommendations
    
    def score(self, user_id: str, item_ids: List[str]) -> Dict[str, float]:
        """
        Score specific user-item pairs.
        
        Args:
            user_id: User ID
            item_ids: List of item IDs to score
            
        Returns:
            Dictionary mapping item_id to score
        """
        if user_id not in self.user_vocab:
            raise ValueError(f"User {user_id} not found")
        
        # Get user embedding
        try:
            user_emb = self.model({'user_id': tf.constant([user_id])}, training=False)['user_embedding']
            user_emb = user_emb.numpy()[0]
        except:
            user_emb = self.model(
                {'user_id': tf.constant([user_id]), 'movie_id': tf.constant([self.item_vocab[0]])}, 
                training=False
            )['user_embedding'].numpy()[0]
        
        # Get item embeddings
        valid_items = [item_id for item_id in item_ids if item_id in self.item_vocab]
        
        if not valid_items:
            raise ValueError("No valid items found")
        
        try:
            item_embs = self.model({'movie_id': tf.constant(valid_items)}, training=False)['item_embedding']
            item_embs = item_embs.numpy()
        except:
            dummy_users = [user_id] * len(valid_items)
            item_embs = self.model(
                {'user_id': tf.constant(dummy_users), 'movie_id': tf.constant(valid_items)}, 
                training=False
            )['item_embedding'].numpy()
        
        # Compute scores
        scores = {}
        for item_id, item_emb in zip(valid_items, item_embs):
            score = float(np.dot(user_emb, item_emb))
            scores[item_id] = score
        
        return scores
    
    def recommend_batch(self, user_ids: List[str], k: int = 10) -> List[Dict]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            k: Number of recommendations per user
            
        Returns:
            List of recommendation sets for each user
        """
        results = []
        for user_id in user_ids:
            try:
                recs = self.recommend(user_id, k)
                results.append({
                    "user_id": user_id,
                    "recommendations": recs,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "user_id": user_id,
                    "recommendations": [],
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    def _get_popular_items(self, k: int) -> List[Dict]:
        """Fallback: return popular items for cold-start users."""
        # Simple popularity: return first K items (in practice, use actual popularity)
        popular_items = []
        for rank, item_id in enumerate(self.item_vocab[:k], 1):
            popular_items.append({
                "item_id": item_id,
                "score": 1.0 / rank,  # Decreasing popularity score
                "rank": rank
            })
        return popular_items
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "version": self.version,
            "num_users": len(self.user_vocab) if self.user_vocab else 0,
            "num_items": len(self.item_vocab) if self.item_vocab else 0,
            "embedding_dim": self.item_embeddings.shape[1] if self.item_embeddings is not None else 0,
            "config": self.config,
            "model_path": str(self.model_dir)
        }
