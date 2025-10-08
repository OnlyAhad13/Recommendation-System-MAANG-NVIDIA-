"""
Data preprocessing pipeline for MovieLens dataset.
"""

import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path


def preprocessing_pipeline(ratings, movies, users, rating_threshold=3):
    """
    Complete preprocessing pipeline for MovieLens data.
    
    Args:
        ratings: DataFrame with user ratings
        movies: DataFrame with movie information
        users: DataFrame with user demographics
        rating_threshold: Threshold for implicit feedback (default: 4)
    
    Returns:
        Dictionary with processed data and metadata
    """
    # Timestamp features
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['rating_year'] = ratings['datetime'].dt.year

    # Sequential IDs for embeddings
    user_id_map = {uid: i for i, uid in enumerate(ratings['user_id'].unique())}
    movie_id_map = {mid: i for i, mid in enumerate(ratings['movie_id'].unique())}
    ratings['user_id'] = ratings['user_id'].map(user_id_map)
    ratings['movie_id'] = ratings['movie_id'].map(movie_id_map)
    movies['movie_id'] = movies['movie_id'].map(movie_id_map)
    users['user_id'] = users['user_id'].map(user_id_map)

    # Drop NaN values
    movies = movies.dropna(subset=['movie_id'])
    movies = movies.copy()
    
    # Movie genres (multi-hot encoding)
    all_genres = set()
    for g in movies['genres']:
        all_genres.update(g.split('|'))
    genre_list = sorted(list(all_genres))
    for genre in genre_list:
        movies[f"genre_{genre}"] = movies['genres'].apply(lambda x: int(genre in x.split('|')))

    # User demographics
    users['age_id'] = pd.Categorical(users['age']).codes
    users['gender_id'] = pd.Categorical(users['gender']).codes
    users['occupation_id'] = pd.Categorical(users['occupation']).codes

    # Extract movie year from title
    year_re = re.compile(r'\((\d{4})\)$')
    def extract_year(title):
        m = year_re.search(title)
        return int(m.group(1)) if m else np.nan

    movies['year'] = movies['title'].apply(extract_year)
    movies['year'] = movies['year'].fillna(movies['year'].median())

    # Normalize release year
    movies['year_normalized'] = (movies['year'] - movies['year'].min()) / (movies['year'].max() - movies['year'].min())
    
    # Merge with ratings for age_at_rating
    ratings = ratings.merge(movies[['movie_id', 'year']], on='movie_id', how='left')
    ratings['age_at_rating'] = (ratings['rating_year'] - ratings['year']).clip(lower=0)

    # Train/val/test temporal split
    ratings_sorted = ratings.sort_values('timestamp')
    n = len(ratings_sorted)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    train_ratings = ratings_sorted.iloc[:train_size].copy()
    val_ratings = ratings_sorted.iloc[train_size:train_size+val_size].copy()
    test_ratings = ratings_sorted.iloc[train_size+val_size:].copy()
    print(f"Temporal split → Train: {len(train_ratings)}, Val: {len(val_ratings)}, Test: {len(test_ratings)}")

    # User and movie aggregates (train-only to prevent leakage)
    train_user_stats = train_ratings.groupby('user_id')['rating'].agg(['mean', 'count', 'std']).reset_index()
    train_user_stats.rename(columns={'mean':'user_avg_rating', 'count':'user_rating_count', 'std':'user_rating_std'}, inplace=True)
    train_user_stats['user_rating_std'] = train_user_stats['user_rating_std'].fillna(0)

    train_movie_stats = train_ratings.groupby('movie_id')['rating'].agg(['mean','count','std']).reset_index()
    train_movie_stats.rename(columns={'mean':'movie_avg_rating','count':'movie_rating_count','std':'movie_rating_std'}, inplace=True)
    train_movie_stats['movie_rating_std'] = train_movie_stats['movie_rating_std'].fillna(0)

    def join_stats(df):
        df = df.merge(train_user_stats, on='user_id', how='left')
        df = df.merge(train_movie_stats, on='movie_id', how='left')
        return df

    train_ratings = join_stats(train_ratings)
    val_ratings = join_stats(val_ratings)
    test_ratings = join_stats(test_ratings)

    # Implicit target (for ranking/retrieval)
    for df in [train_ratings, val_ratings, test_ratings]:
        df['y_implicit'] = (df['rating'] >= rating_threshold).astype(int)

    # Feature dictionaries
    user_features = users.set_index('user_id')[['age_id', 'gender_id', 'occupation_id']].to_dict('index')
    movie_features = movies.set_index('movie_id')[
        [c for c in movies.columns if c.startswith('genre_')] + ['year_normalized']
    ].to_dict('index')

    return {
        'train_ratings': train_ratings,
        'val_ratings': val_ratings,
        'test_ratings': test_ratings,
        'user_features': user_features,
        'movie_features': movie_features,
        'train_user_stats': train_user_stats,
        'train_movie_stats': train_movie_stats,
        'n_users': len(user_id_map),
        'n_movies': len(movie_id_map),
        'genre_list': genre_list,
        'user_id_map': user_id_map,
        'movie_id_map': movie_id_map
    }


def check_preprocessing_quality(processed_data):
    """Validate the quality of preprocessed data."""
    print("\n=== PREPROCESSING QUALITY CHECK ===")

    train = processed_data['train_ratings']

    print(f"User ID range: 0–{train['user_id'].max()} (expected {processed_data['n_users']-1})")
    print(f"Movie ID range: 0–{train['movie_id'].max()} (expected {processed_data['n_movies']-1})")

    print(f"Missing values in train: {train.isnull().sum().sum()}")
    print(f"Data types – user_id: {train['user_id'].dtype}, movie_id: {train['movie_id'].dtype}")

    all_users_train = set(train['user_id'])
    all_movies_train = set(train['movie_id'])
    val_users = set(processed_data['val_ratings']['user_id'])
    test_users = set(processed_data['test_ratings']['user_id'])

    print(f"Val cold-start users: {len(val_users - all_users_train)}")
    print(f"Test cold-start users: {len(test_users - all_users_train)}")


def load_movielens_data(data_dir: str = None):
    """
    Load MovieLens dataset from files.
    
    Args:
        data_dir: Path to directory containing MovieLens .dat files
                 If None, uses default path: ../data/raw/
    
    Returns:
        Tuple of (ratings, movies, users) DataFrames
    """
    if data_dir is None:
        # Default to data/raw directory relative to project root
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    else:
        data_dir = Path(data_dir)
    
    print(f"Loading data from: {data_dir}")
    
    ratings = pd.read_csv(
        data_dir / 'ratings.dat',
        sep="::", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine='python'
    )

    movies = pd.read_csv(
        data_dir / 'movies.dat',
        sep="::", header=None,
        names=["movie_id", "title", "genres"],
        engine='python',
        encoding='latin-1'
    )

    users = pd.read_csv(
        data_dir / 'users.dat',
        sep="::", header=None,
        names=["user_id", "gender", "age", "occupation", "zip-code"],
        engine='python'
    )
    
    return ratings, movies, users


def main(data_dir: str = None, output_path: str = None):
    """
    Main preprocessing function.
    
    Args:
        data_dir: Path to raw data directory
        output_path: Path to save processed data pickle file
    """
    # Load data
    ratings, movies, users = load_movielens_data(data_dir)
    
    # Preprocess
    processed = preprocessing_pipeline(ratings, movies, users)
    
    # Quality check
    check_preprocessing_quality(processed)
    
    # Save processed data
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'processed_data.pkl'
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(processed, f)
    
    print(f"\n✓ Preprocessing complete! Saved to: {output_path}")
    print(f"  - Train samples: {len(processed['train_ratings'])}")
    print(f"  - Val samples: {len(processed['val_ratings'])}")
    print(f"  - Test samples: {len(processed['test_ratings'])}")
    print(f"  - Users: {processed['n_users']}")
    print(f"  - Movies: {processed['n_movies']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess MovieLens dataset')
    parser.add_argument('--data_dir', type=str, default=None, 
                       help='Path to directory containing .dat files')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save processed data pickle file')
    
    args = parser.parse_args()
    main(args.data_dir, args.output)

