"""
Simple model loader that bypasses custom class registration issues.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class SimpleRecommendationService:
    """Simplified recommendation service that loads models without custom classes."""
    
    def __init__(self, model_dir: str = "outputs/models/experiment_001"):
        self.model_dir = Path(model_dir)
        self.user_vocab = None
        self.item_vocab = None
        self.item_embeddings = None
        self.config = None
        self.version = "1.0.0"
        
    def load_model(self):
        """Load vocabularies and create a simple recommendation system."""
        logger.info(f"Loading model from {self.model_dir}")
        
        try:
            # Load vocabularies
            vocab_path = self.model_dir / "vocabs.json"
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    vocabs = json.load(f)
                    self.user_vocab = vocabs['users']
                    self.item_vocab = vocabs['items']
                logger.info(f"✅ Vocabularies loaded: {len(self.user_vocab)} users, {len(self.item_vocab)} items")
            else:
                raise FileNotFoundError(f"Vocabularies not found at {vocab_path}")
            
            # Load config
            config_path = self.model_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("✅ Config loaded")
            
            # Create simple embeddings (random for now - in production, load from model)
            embedding_dim = self.config.get('embedding_dim', 64) if self.config else 64
            self.item_embeddings = np.random.randn(len(self.item_vocab), embedding_dim)
            
            logger.info("✅ Simple model service ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return (self.user_vocab is not None and 
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
        
        # For now, return random recommendations (in production, use actual model)
        import random
        random.seed(hash(user_id))  # Deterministic randomness based on user_id
        
        # Get random subset of items
        num_items = min(k, len(self.item_vocab))
        selected_indices = random.sample(range(len(self.item_vocab)), num_items)
        
        # Format results
        recommendations = []
        for rank, idx in enumerate(selected_indices, 1):
            recommendations.append({
                "item_id": self.item_vocab[idx],
                "score": float(np.random.rand()),  # Random score
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
        
        # For now, return random scores (in production, use actual model)
        scores = {}
        for item_id in item_ids:
            if item_id in self.item_vocab:
                scores[item_id] = float(np.random.rand())
        
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
