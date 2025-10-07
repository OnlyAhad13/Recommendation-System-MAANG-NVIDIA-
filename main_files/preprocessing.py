import pandas as pd
import numpy as np
import re
import pickle

def preprocessing_pipeline(ratings, movies, users, rating_threshold=4):
    #Timestamp
    ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s' )
    ratings['rating_year'] = ratings['datetime'].dt.year

    #Sequential IDs for embeddings
    user_id_map = {uid: i for i, uid in enumerate(ratings['user_id'].unique())}
    movie_id_map = {mid: i for i, mid in enumerate(ratings['movie_id'].unique())}
    ratings['user_id'] = ratings['user_id'].map(user_id_map)
    ratings['movie_id'] = ratings['movie_id'].map(movie_id_map)
    movies['movie_id'] = movies['movie_id'].map(movie_id_map)
    users['user_id'] = users['user_id'].map(user_id_map)

    #Dropping for NaN
    movies = movies.dropna(subset=['movie_id'])
    movies = movies.copy()
    #Movie genres(Multi hot)
    all_genres = set()
    for g in movies['genres']:
        all_genres.update(g.split('|'))
    genre_list = sorted(list(all_genres))
    for genre in genre_list:
        movies[f"genre_{genre}"] = movies['genres'].apply(lambda x: int(genre in x.split('|')))

    #User demographics
    users['age_id'] = pd.Categorical(users['age']).codes
    users['gender_id'] = pd.Categorical(users['gender']).codes
    users['occupation_id'] = pd.Categorical(users['occupation']).codes

    #Extracting movie year and age_at_rating
    year_re = re.compile(r'\((\d{4})\)$')
    def extract_year(title):
        m = year_re.search(title)
        return int(m.group(1)) if m else np.nan

    movies['year'] = movies['title'].apply(extract_year)
    movies['year'] = movies['year'].fillna(movies['year'].median())

    #Normalizing release year for model input
    movies['year_normalized'] = (movies['year'] - movies['year'].min()) / (movies['year'].max() - movies['year'].min())
    #Merging with ratings for at_age_rating
    ratings = ratings.merge(movies[['movie_id', 'year']], on='movie_id', how='left')
    ratings['age_at_rating'] = (ratings['rating_year'] - ratings['year']).clip(lower=0)

    #Train/val/test temporal split
    ratings_sorted = ratings.sort_values('timestamp')
    n = len(ratings_sorted)
    train_size = int(0.8*n)
    val_size = int(0.1*n)
    train_ratings = ratings_sorted.iloc[:train_size].copy()
    val_ratings = ratings_sorted.iloc[train_size:train_size+val_size].copy()
    test_ratings = ratings_sorted.iloc[train_size+val_size:].copy()
    print(f"Temporal split → Train: {len(train_ratings)}, Val: {len(val_ratings)}, Test: {len(test_ratings)}")

    #Aggregates (train-only)
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

    #Implicit Target (for ranking/retrieval)
    for df in [train_ratings, val_ratings, test_ratings]:
        df['y_implicit'] = (df['rating'] >= rating_threshold).astype(int)

    #Feature dicts
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

#Loading User Data
def load_data():
    ratings = pd.read_csv('../movies_dataset/ratings.dat',
                sep="::", header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
                engine='python')

    movies = pd.read_csv('../movies_dataset/movies.dat',
                sep="::", header=None,
                names=["movie_id", "title", "genres"],
                engine='python',
                encoding='latin-1')

    users = pd.read_csv('../movies_dataset/users.dat',
                sep="::", header=None,
                names=["user_id", "gender", "age", "occupation", "zip-code"],
                engine='python')
    return ratings, movies, users

ratings, movies, users = load_data()
processed = preprocessing_pipeline(ratings, movies, users)
check_preprocessing_quality(processed)

with open("../movies_dataset/processed_data.pkl", "wb") as f:
    pickle.dump(processed, f)