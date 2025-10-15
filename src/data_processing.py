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
    
    def __init__(self, config):
        self.config = config
        self.feature_scalers = {}
        # Cache for train-only statistics (prevent leakage)
        self.train_user_genre_prefs = None
        self.train_user_temporal_stats = None
        self.train_item_temporal_stats = None
        
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
        """Engineer features from raw data with advanced signals."""
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
        
        # === BASIC TIME FEATURES ===
        if 'timestamp' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df['hour'] = df['datetime'].dt.hour
                df['day_of_week'] = df['datetime'].dt.dayofweek
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                df['month'] = df['datetime'].dt.month
            except Exception as e:
                logger.warning(f"Failed to parse timestamps: {e}")
        
        # === USER STATISTICS ===
        try:
            if 'rating' in df.columns:
                user_stats = df.groupby('user_id').agg({
                    'rating': ['count', 'mean', 'std', 'min', 'max'], 
                    'movie_id': 'nunique'
                }).fillna(0)
                user_stats.columns = ['user_rating_count', 'user_avg_rating', 
                                     'user_rating_std', 'user_min_rating', 
                                     'user_max_rating', 'user_unique_items']
                # User rating range (how diverse are their ratings?)
                user_stats['user_rating_range'] = user_stats['user_max_rating'] - user_stats['user_min_rating']
            else:
                user_stats = df.groupby('user_id').agg({'movie_id': ['count', 'nunique']}).fillna(0)
                user_stats.columns = ['user_rating_count', 'user_unique_items']
            
            df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
        except Exception as e:
            logger.warning(f"Failed to compute user stats: {e}")
        
        # === ITEM STATISTICS ===
        try:
            if 'rating' in df.columns:
                item_stats = df.groupby('movie_id').agg({
                    'rating': ['count', 'mean', 'std', 'min', 'max'], 
                    'user_id': 'nunique'
                }).fillna(0)
                item_stats.columns = ['item_rating_count', 'item_avg_rating', 
                                     'item_rating_std', 'item_min_rating',
                                     'item_max_rating', 'item_unique_users']
                # Item rating polarization
                item_stats['item_rating_range'] = item_stats['item_max_rating'] - item_stats['item_min_rating']
            else:
                item_stats = df.groupby('movie_id').agg({'user_id': ['count', 'nunique']}).fillna(0)
                item_stats.columns = ['item_rating_count', 'item_unique_users']
            
            df = df.merge(item_stats, left_on='movie_id', right_index=True, how='left')
        except Exception as e:
            logger.warning(f"Failed to compute item stats: {e}")
        
        # === ADVANCED TEMPORAL FEATURES ===
        if 'timestamp' in df.columns and 'datetime' in df.columns:
            try:
                # User recency signals
                user_temporal = df.groupby('user_id')['datetime'].agg(['min', 'max', 'count'])
                user_temporal['user_activity_days'] = (user_temporal['max'] - user_temporal['min']).dt.days + 1
                user_temporal['user_rating_velocity'] = user_temporal['count'] / user_temporal['user_activity_days'].replace(0, 1)
                
                # Days since user first/last active
                df = df.merge(user_temporal[['min', 'max']], left_on='user_id', right_index=True, how='left', suffixes=('', '_user'))
                df['days_since_user_first_active'] = (df['datetime'] - df['min']).dt.days
                df['days_since_user_last_active'] = (df['datetime'].max() - df['max']).dt.days
                df = df.drop(['min', 'max'], axis=1, errors='ignore')
                
                # Merge velocity
                df = df.merge(user_temporal[['user_rating_velocity']], left_on='user_id', right_index=True, how='left')
                
                # Item temporal patterns
                item_temporal = df.groupby('movie_id')['datetime'].agg(['min', 'max', 'count'])
                item_temporal['item_age_days'] = (df['datetime'].max() - item_temporal['min']).dt.days
                
                # Recent popularity (last 90 days)
                recent_cutoff = df['datetime'].max() - pd.Timedelta(days=90)
                recent_item_counts = df[df['datetime'] > recent_cutoff].groupby('movie_id').size()
                item_temporal['item_recent_popularity'] = recent_item_counts
                item_temporal['item_recent_popularity'] = item_temporal['item_recent_popularity'].fillna(0)
                item_temporal['item_popularity_trend'] = item_temporal['item_recent_popularity'] / item_temporal['count'].replace(0, 1)
                
                df = df.merge(item_temporal[['item_age_days', 'item_popularity_trend']], 
                             left_on='movie_id', right_index=True, how='left')
                
                # Cache for val/test
                if mode == 'train':
                    self.train_user_temporal_stats = user_temporal
                    self.train_item_temporal_stats = item_temporal
                    
            except Exception as e:
                logger.warning(f"Failed to compute temporal features: {e}")
        
        # === USER-ITEM INTERACTION FEATURES ===
        try:
            # User rating consistency (inverse of std)
            if 'user_rating_std' in df.columns:
                df['user_rating_consistency'] = 1.0 / (1.0 + df['user_rating_std'])
            
            # Item rating polarization (high std = controversial)
            if 'item_rating_std' in df.columns:
                df['item_polarization'] = df['item_rating_std']
            
            # User-item popularity match
            if 'user_rating_count' in df.columns and 'item_rating_count' in df.columns:
                df['user_popularity_match'] = np.abs(
                    np.log1p(df['user_rating_count']) - np.log1p(df['item_rating_count'])
                )
        except Exception as e:
            logger.warning(f"Failed to compute interaction features: {e}")
        
        # === GENRE PREFERENCE FEATURES ===
        if isinstance(item_features, pd.DataFrame) and not item_features.empty:
            try:
                # Get genre columns
                genre_cols = [c for c in item_features.columns if c.startswith('genre_')]
                
                if genre_cols and mode == 'train':
                    # Compute user genre preferences from training data only
                    item_genres = item_features[['movie_id'] + genre_cols].copy()
                    item_genres['movie_id'] = item_genres['movie_id'].astype(str)
                    
                    df_with_genres = df.merge(item_genres, on='movie_id', how='left')
                    
                    if 'rating' in df_with_genres.columns:
                        # Weighted genre preference (higher ratings = stronger preference)
                        user_genre_ratings = df_with_genres.groupby('user_id').apply(
                            lambda x: pd.Series({
                                col: (x[col] * x['rating']).sum() / x['rating'].sum() if x['rating'].sum() > 0 else 0
                                for col in genre_cols
                            })
                        )
                    else:
                        # Simple genre preference (binary)
                        user_genre_ratings = df_with_genres.groupby('user_id')[genre_cols].mean()
                    
                    self.train_user_genre_prefs = user_genre_ratings
                
                # Apply genre preferences
                if self.train_user_genre_prefs is not None and genre_cols:
                    item_genres = item_features[['movie_id'] + genre_cols].copy()
                    item_genres['movie_id'] = item_genres['movie_id'].astype(str)
                    df = df.merge(item_genres, on='movie_id', how='left', suffixes=('', '_item'))
                    
                    # Calculate user-item genre match
                    def compute_genre_match(row):
                        user_id = row['user_id']
                        if user_id not in self.train_user_genre_prefs.index:
                            return 0.0
                        
                        user_prefs = self.train_user_genre_prefs.loc[user_id]
                        item_genres_vec = row[[c for c in genre_cols if c in row.index]].values
                        
                        # Dot product of user preferences and item genres
                        match_score = np.dot(user_prefs.values, item_genres_vec)
                        return match_score
                    
                    df['user_genre_match'] = df.apply(compute_genre_match, axis=1)
                    
                    # Drop genre columns after computing match
                    df = df.drop(columns=[c for c in genre_cols if c in df.columns], errors='ignore')
                    
            except Exception as e:
                logger.warning(f"Failed to compute genre features: {e}")
        
        # === MERGE EXTERNAL FEATURES ===
        for features, key_col, id_col in [(user_features, 0, 'user_id'), 
                                           (item_features, 0, 'movie_id')]:
            if isinstance(features, pd.DataFrame) and not features.empty:
                try:
                    feat_copy = features.copy()
                    # Exclude genre columns from item_features (already processed)
                    if id_col == 'movie_id':
                        genre_cols = [c for c in feat_copy.columns if c.startswith('genre_')]
                        feat_copy = feat_copy.drop(columns=genre_cols, errors='ignore')
                    
                    feat_copy.iloc[:, key_col] = feat_copy.iloc[:, key_col].astype(str)
                    df = df.merge(feat_copy, left_on=id_col, 
                                right_on=feat_copy.columns[key_col], 
                                how='left', suffixes=('', '_ext'))
                except Exception as e:
                    logger.warning(f"Failed to merge external features: {e}")
        
        # === SCALE NUMERICAL FEATURES ===
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in ['user_id', 'movie_id', 'rating', 'y_implicit', 'timestamp']]
        
        if numeric_cols:
            try:
                if mode == 'train':
                    self.feature_scalers['numeric'] = StandardScaler()
                    df[numeric_cols] = self.feature_scalers['numeric'].fit_transform(df[numeric_cols])
                    logger.info(f"Scaled {len(numeric_cols)} numerical features")
                elif 'numeric' in self.feature_scalers:
                    df[numeric_cols] = self.feature_scalers['numeric'].transform(df[numeric_cols])
            except Exception as e:
                logger.warning(f"Failed to scale features: {e}")
        
        # Drop datetime column (not needed for model)
        df = df.drop(['datetime'], axis=1, errors='ignore')
        
        logger.info(f"Final feature count for {mode}: {len(df.columns)} columns")
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

