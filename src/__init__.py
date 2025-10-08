"""
Enterprise Recommendation System

A production-ready recommendation system using two-tower retrieval 
and Deep & Cross Network (DCN) ranking models.
"""

__version__ = "1.0.0"

from .config import ModelConfig
from .trainer import ProductionTrainer

__all__ = ['ModelConfig', 'ProductionTrainer']

