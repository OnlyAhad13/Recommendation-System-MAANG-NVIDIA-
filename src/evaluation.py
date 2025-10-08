"""
Evaluation metrics and callbacks for the recommendation system.
"""

import time
import json
import logging
from pathlib import Path
from typing import List
import numpy as np
from tensorflow import keras

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Advanced evaluation metrics for recommendation systems."""
    
    @staticmethod
    def recall_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
        """Calculate Recall@K."""
        recalls = []
        for pred, true in zip(predictions, ground_truth):
            top_k = pred[:k]
            recalls.append(1.0 if true in top_k else 0.0)
        return np.mean(recalls) if recalls else 0.0
    
    @staticmethod
    def precision_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
        """Calculate Precision@K."""
        precisions = []
        for pred, true in zip(predictions, ground_truth):
            top_k = pred[:k]
            precisions.append(1.0 / len(top_k) if true in top_k else 0.0)
        return np.mean(precisions) if precisions else 0.0
    
    @staticmethod
    def ndcg_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
        """Calculate NDCG@K."""
        ndcgs = []
        for pred, true in zip(predictions, ground_truth):
            top_k = pred[:k]
            if true in top_k:
                rank = top_k.index(true) + 1
                dcg = 1.0 / np.log2(rank + 1)
                idcg = 1.0 / np.log2(2)  # Best possible rank is 1
                ndcgs.append(dcg / idcg)
            else:
                ndcgs.append(0.0)
        return np.mean(ndcgs) if ndcgs else 0.0
    
    @staticmethod
    def map_at_k(predictions: List[List[str]], ground_truth: List[str], k: int) -> float:
        """Calculate MAP@K (Mean Average Precision)."""
        aps = []
        for pred, true in zip(predictions, ground_truth):
            top_k = pred[:k]
            if true in top_k:
                rank = top_k.index(true) + 1
                aps.append(1.0 / rank)
            else:
                aps.append(0.0)
        return np.mean(aps) if aps else 0.0
    
    @staticmethod
    def mrr(predictions: List[List[str]], ground_truth: List[str]) -> float:
        """Calculate MRR (Mean Reciprocal Rank)."""
        rrs = []
        for pred, true in zip(predictions, ground_truth):
            if true in pred:
                rank = pred.index(true) + 1
                rrs.append(1.0 / rank)
            else:
                rrs.append(0.0)
        return np.mean(rrs) if rrs else 0.0
    
    @staticmethod
    def coverage(recommendations: List[List[str]], all_items: List[str]) -> float:
        """Calculate catalog coverage."""
        if not all_items:
            return 0.0
        recommended = set()
        for rec in recommendations:
            recommended.update(rec)
        return len(recommended) / len(all_items)
    
    @staticmethod
    def diversity(recommendations: List[List[str]]) -> float:
        """Calculate diversity of recommendations."""
        if not recommendations:
            return 0.0
        diversities = []
        for rec in recommendations:
            if len(rec) > 1:
                diversities.append(len(set(rec)) / len(rec))
            else:
                diversities.append(0.0)
        return np.mean(diversities)


class CustomMetricsCallback(keras.callbacks.Callback):
    """Custom callback to track detailed metrics during training."""
    
    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir
        self.history = []
        self.start_time = time.time()
        self.epoch_start = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = time.time() - self.epoch_start
        
        # System metrics
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                cpu = psutil.cpu_percent(0.1)
            except:
                mem, cpu = 0, 0
        else:
            mem, cpu = 0, 0
        
        metrics = {
            'epoch': epoch,
            'time': time.time(),
            'epoch_seconds': epoch_time,
            'memory_mb': mem,
            'cpu_pct': cpu,
            **{k: float(v) for k, v in logs.items()}
        }
        
        self.history.append(metrics)
        logger.info(f"Epoch {epoch+1}: {epoch_time:.1f}s | Loss: {logs.get('loss', 0):.4f} | Mem: {mem:.0f}MB")
        
        # Save metrics periodically
        if (epoch + 1) % 2 == 0:
            try:
                with open(self.output_dir / 'detailed_metrics.json', 'w') as f:
                    json.dump(self.history, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save metrics: {e}")
    
    def on_train_end(self, logs=None):
        """Save final metrics."""
        try:
            with open(self.output_dir / 'detailed_metrics.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save final metrics: {e}")

