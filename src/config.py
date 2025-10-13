"""
Configuration classes for the recommendation system.
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class ModelConfig:
    """Configuration for the recommendation model."""
    
    # Embedding dimensions (smaller for easier retrieval learning)
    embedding_dim: int = 64 
    user_tower_dims: List[int] = None
    item_tower_dims: List[int] = None
    
    # DCN parameters (simpler for retrieval focus)
    cross_layers: int = 1 
    dnn_dims: List[int] = None
    dropout_rate: float = 0.3
    l2_reg: float = 1e-4
    
    # Training parameters
    batch_size: int = 2048  # Smaller for better gradients
    learning_rate_retrieval: float = 0.01  # Higher for retrieval focus
    learning_rate_ranking: float = 0.0001
    epochs_retrieval: int = 20  # More epochs for retrieval to learn
    epochs_ranking: int = 5
    warmup_steps: int = 1000
    
    # Negative sampling (more negatives = better retrieval)
    num_hard_negatives: int = 50  # Increased
    num_random_negatives: int = 50  # Increased
    negative_sampling_strategy: str = "mixed"  # Mix is better
    
    # Multi-task weights (retrieval implicitly has weight 1.0)
    retrieval_weight: float = 10.0  # Pure retrieval
    ctr_weight: float = 0.2  
    rating_weight: float = 0.2
    
    # Evaluation
    eval_topk: List[int] = None
    
    # System settings
    mixed_precision: bool = True
    distributed_strategy: str = "none"
    
    def __post_init__(self):
        if self.user_tower_dims is None:
            self.user_tower_dims = [64]  # Simpler tower
        if self.item_tower_dims is None:
            self.item_tower_dims = [64]  # Simpler tower
        if self.dnn_dims is None:
            self.dnn_dims = [64]  # Much simpler DCN
        if self.eval_topk is None:
            self.eval_topk = [5, 10, 20, 50]
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)

