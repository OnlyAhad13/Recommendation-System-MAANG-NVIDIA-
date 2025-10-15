# app/recommendation_service.py

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf
import faiss

# Now we only need to import the custom classes themselves
from src.models import MultiTowerModel, DeepCrossNetwork, MultiTaskModel

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    A professional, production-ready recommendation service that uses a
    trained TensorFlow model and a FAISS index for fast retrieval.
    """

    def __init__(self, model_dir: str = "outputs/models/experiment_001"):
        self.model_dir = Path(model_dir)
        self.version = "1.2.0" # Updated version

        # Model artifacts, initialized to None
        self.user_vocab: List[str] = []
        self.item_map: Dict[str, str] = {}
        self.encoder_model: tf.keras.Model = None
        self.faiss_index: faiss.Index = None

    def load(self):
        """Loads all necessary artifacts for the model to serve predictions."""
        logger.info(f"Loading model artifacts from {self.model_dir}...")

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # 1. Load vocabularies
        with open(self.model_dir / "vocabs.json", 'r') as f:
            self.user_vocab = json.load(f)['users']
        logger.info(f"Loaded {len(self.user_vocab)} users from vocab.")

        # 2. Load FAISS index and item map
        self.faiss_index = faiss.read_index(str(self.model_dir / "faiss.idx"))
        with open(self.model_dir / "item_map.json", 'r') as f:
            self.item_map = json.load(f)
        logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} items.")

        # 3. Load the TensorFlow model with a simple custom_objects dictionary
        model_path = self.model_dir / "encoder.keras"
        custom_objects = {
            "MultiTowerModel": MultiTowerModel,
            "DeepCrossNetwork": DeepCrossNetwork
        }
        self.encoder_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("✅ TensorFlow encoder model loaded successfully.")

    def is_ready(self) -> bool:
        return all([self.encoder_model, self.faiss_index, self.user_vocab, self.item_map])

    def recommend(self, user_id: str, k: int = 10) -> List[Dict]:
        if user_id not in self.user_vocab:
            logger.warning(f"User '{user_id}' not in vocabulary. Applying cold-start strategy.")
            return self._get_popular_items(k)
        user_tensor = tf.constant([user_id])
        user_embedding = self.encoder_model({"user_id": user_tensor}, training=False)['user_embedding']
        user_embedding_np = user_embedding.numpy().astype('float32')
        faiss.normalize_L2(user_embedding_np)
        distances, indices = self.faiss_index.search(user_embedding_np, k)
        recommendations = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                recommendations.append({
                    "item_id": self.item_map[str(idx)],
                    "score": float(distances[0][i]),
                    "rank": i + 1
                })
        return recommendations

    def score(self, user_id: str, item_ids: List[str]) -> Dict[str, float]:
        if user_id not in self.user_vocab:
            raise ValueError(f"User '{user_id}' not found in vocabulary.")
        user_tensor = tf.constant([user_id])
        item_tensor = tf.constant(item_ids)
        embeddings = self.encoder_model({"user_id": user_tensor, "movie_id": item_tensor}, training=False)
        user_embedding = embeddings['user_embedding']
        item_embeddings = embeddings['item_embedding']
        scores = tf.linalg.matvec(item_embeddings, user_embedding[0])
        return {item_id: float(score) for item_id, score in zip(item_ids, scores.numpy())}

    def _get_popular_items(self, k: int) -> List[Dict]:
        popular_items = []
        for i in range(min(k, len(self.item_map))):
            popular_items.append({
                "item_id": self.item_map[str(i)],
                "score": 1.0 - (i * 0.05),
                "rank": i + 1
            })
        return popular_items
        
    def get_model_info(self) -> Dict:
        return {
            "version": self.version,
            "model_path": str(self.model_dir),
            "num_users": len(self.user_vocab),
            "faiss_index_items": self.faiss_index.ntotal if self.faiss_index else 0,
        }