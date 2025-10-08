"""
Configuration classes for the recommendation system.
"""

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class ModelConfig:
    """Configuration for the recommendation model."""
    
    # Embedding dimensions
    embedding_dim: int = 128
    user_tower_dims: List[int] = None
    item_tower_dims: List[int] = None
    
    # DCN parameters
    cross_layers: int = 2
    dnn_dims: List[int] = None
    dropout_rate: float = 0.35
    l2_reg: float = 1e-4
    
    # Training parameters
    batch_size: int = 4096
    learning_rate_retrieval: float = 0.005
    learning_rate_ranking: float = 0.0001
    epochs_retrieval: int = 10
    epochs_ranking: int = 5
    warmup_steps: int = 1000
    
    # Negative sampling
    num_hard_negatives: int = 20
    num_random_negatives: int = 30
    negative_sampling_strategy: str = "hard"
    
    # Multi-task weights
    ctr_weight: float = 0.5
    rating_weight: float = 0.5
    
    # Evaluation
    eval_topk: List[int] = None
    
    # System settings
    mixed_precision: bool = True
    distributed_strategy: str = "none"
    
    def __post_init__(self):
        if self.user_tower_dims is None:
            self.user_tower_dims = [128, 64]
        if self.item_tower_dims is None:
            self.item_tower_dims = [128, 64]
        if self.dnn_dims is None:
            self.dnn_dims = [128, 64, 32]
        if self.eval_topk is None:
            self.eval_topk = [5, 10, 20, 50]
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)

