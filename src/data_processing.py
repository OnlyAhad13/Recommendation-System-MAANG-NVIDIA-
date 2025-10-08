"""
Data processing and feature engineering for the recommendation system.
"""

import logging
from typing import Dict, Any, List, Set
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, validation, and feature engineering."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_scalers = {}
        
    def load_and_validate_data(self, pickle_path: str) -> Dict[str, Any]:
        """Load data from pickle file and validate structure."""
        logger.info(f"Loading data from {pickle_path}")
        
        try:
            import pickle
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
        except:
            data = pd.read_pickle(pickle_path)
        
        if isinstance(data, dict):
            train_df = data.get('train_ratings', data.get('train', pd.DataFrame()))
            val_df = data.get('val_ratings', data.get('val', pd.DataFrame()))
            test_df = data.get('test_ratings', data.get('test', pd.DataFrame()))
            user_features = data.get('user_features', pd.DataFrame())
            item_features = data.get('movie_features', data.get('item_features', pd.DataFrame()))
        else:
            train_df, val_df, test_df = data, pd.DataFrame(), pd.DataFrame()
            user_features, item_features = pd.DataFrame(), pd.DataFrame()
        
        return {
            'train_df': pd.DataFrame(train_df) if not isinstance(train_df, pd.DataFrame) else train_df,
            'val_df': pd.DataFrame(val_df) if not isinstance(val_df, pd.DataFrame) else val_df,
            'test_df': pd.DataFrame(test_df) if not isinstance(test_df, pd.DataFrame) else test_df,
            'user_features': user_features if isinstance(user_features, pd.DataFrame) else pd.DataFrame(),
            'item_features': item_features if isinstance(item_features, pd.DataFrame) else pd.DataFrame()
        }
    
    def engineer_features(self, df: pd.DataFrame, user_features: pd.DataFrame, 
                         item_features: pd.DataFrame, mode: str = 'train') -> pd.DataFrame:
        """Engineer features from raw data."""
        if df.empty:
            return df
        
        logger.info(f"Engineering features for {mode} - Shape: {df.shape}")
        df = df.copy()
        
        # Ensure required columns exist
        for col, alternatives in [('user_id', ['user', 'userid', 'UserID']), 
                                  ('movie_id', ['movie', 'movieid', 'item_id', 'MovieID'])]:
            if col not in df.columns:
                for alt in alternatives:
                    if alt in df.columns:
                        df.rename(columns={alt: col}, inplace=True)
                        break
                else:
                    raise ValueError(f"Column {col} not found. Available: {df.columns.tolist()}")
        
        # Convert IDs to strings
        df['user_id'] = df['user_id'].astype(str)
        df['movie_id'] = df['movie_id'].astype(str)
        
        # Time features
        if 'timestamp' in df.columns:
            try:
                df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            except:
                pass
        
        # User statistics
        try:
            if 'rating' in df.columns:
                user_stats = df.groupby('user_id').agg({
                    'rating': ['count', 'mean', 'std'], 
                    'movie_id': 'nunique'
                }).fillna(0)
                user_stats.columns = ['user_rating_count', 'user_avg_rating', 
                                     'user_rating_std', 'user_unique_items']
            else:
                user_stats = df.groupby('user_id').agg({'movie_id': ['count', 'nunique']}).fillna(0)
                user_stats.columns = ['user_rating_count', 'user_unique_items']
            df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
        except:
            pass
        
        # Item statistics
        try:
            if 'rating' in df.columns:
                item_stats = df.groupby('movie_id').agg({
                    'rating': ['count', 'mean', 'std'], 
                    'user_id': 'nunique'
                }).fillna(0)
                item_stats.columns = ['item_rating_count', 'item_avg_rating', 
                                     'item_rating_std', 'item_unique_users']
            else:
                item_stats = df.groupby('movie_id').agg({'user_id': ['count', 'nunique']}).fillna(0)
                item_stats.columns = ['item_rating_count', 'item_unique_users']
            df = df.merge(item_stats, left_on='movie_id', right_index=True, how='left')
        except:
            pass
        
        # Merge external features
        for features, key_col, id_col in [(user_features, 0, 'user_id'), 
                                           (item_features, 0, 'movie_id')]:
            if isinstance(features, pd.DataFrame) and not features.empty:
                try:
                    feat_copy = features.copy()
                    feat_copy.iloc[:, key_col] = feat_copy.iloc[:, key_col].astype(str)
                    df = df.merge(feat_copy, left_on=id_col, 
                                right_on=feat_copy.columns[key_col], 
                                how='left', suffixes=('', '_ext'))
                except:
                    pass
        
        # Scale numerical features
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in ['user_id', 'movie_id', 'rating', 'y_implicit', 'timestamp']]
        
        if numeric_cols:
            try:
                if mode == 'train':
                    self.feature_scalers['numeric'] = StandardScaler()
                    df[numeric_cols] = self.feature_scalers['numeric'].fit_transform(df[numeric_cols])
                elif 'numeric' in self.feature_scalers:
                    df[numeric_cols] = self.feature_scalers['numeric'].transform(df[numeric_cols])
            except:
                pass
        
        return df.fillna(0)


class NegativeSampler:
    """Handles negative sampling strategies for training."""
    
    def __init__(self, strategy: str = "mixed", num_hard: int = 5, num_random: int = 50):
        self.strategy = strategy
        self.num_hard = num_hard
        self.num_random = num_random
        self.item_popularity = {}
        self.user_item_matrix = {}
        self.all_items = set()
        
    def fit(self, train_df: pd.DataFrame):
        """Compute statistics from training data."""
        self.item_popularity = train_df.groupby('movie_id').size().to_dict()
        self.user_item_matrix = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()
        self.all_items = set(train_df['movie_id'].unique())
    
    def sample_negatives(self, user_id: str, k: int = None) -> List[str]:
        """Sample negative items for a user."""
        if k is None:
            k = self.num_random
        
        # Get items user hasn't interacted with
        seen_items = self.user_item_matrix.get(user_id, set())
        unseen_items = list(self.all_items - seen_items)
        
        if not unseen_items:
            return []
        
        if self.strategy == "random":
            return list(np.random.choice(unseen_items, min(k, len(unseen_items)), replace=False))
        
        elif self.strategy == "hard":
            # Sample popular items user hasn't seen
            item_scores = [(item, self.item_popularity.get(item, 0)) for item in unseen_items]
            item_scores.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in item_scores[:k]]
        
        elif self.strategy == "mixed":
            # Mix of hard and random negatives
            n_hard = min(self.num_hard, len(unseen_items))
            n_random = min(self.num_random, len(unseen_items) - n_hard)
            
            item_scores = [(item, self.item_popularity.get(item, 0)) for item in unseen_items]
            item_scores.sort(key=lambda x: x[1], reverse=True)
            hard_negs = [item for item, _ in item_scores[:n_hard]]
            
            remaining = [item for item, _ in item_scores[n_hard:]]
            if remaining and n_random > 0:
                random_negs = list(np.random.choice(remaining, n_random, replace=False))
            else:
                random_negs = []
            
            return hard_negs + random_negs
        
        return []

