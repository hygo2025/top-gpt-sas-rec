# Ratins df columns:
idf_user = "user_id"
idf_item = "movie_id"
idf_rating = "rating"
idf_timestamp = "timestamp"
idf_title = "title"
idf_genres = "genres"

# Predictions df columns:
idf_prediction = "prediction"

# Filtering variables
default_k = 10
default_threshold = 10

# other
seed = 42

# Similarity
idf_cosine = "cosine_similarity"
idf_cooccurrence = "cooccurrence"

# Metrics
idf_map = "MAP"
idf_ndcg = "nDCG@K"
idf_precision = "Precision@K"
idf_recall = "Recall@K"
idf_r2 = "R2"
idf_rmse = "RMSE"
idf_mae = "MAE"
idf_exp_var = "Explained Variance"

#imbd url
imdb_url = "https://www.kaggle.com/api/v1/datasets/download/rounakbanik/the-movies-dataset"

def get_imdb_paths(imdb_base_path: str = None):
    if imdb_base_path is None:
        imdb_base_path = '/tmp/imdb'
    return dict({
        'imdb_output_path': f"{imdb_base_path}/movies-dataset.zip",
        'imdb_extract_to': f"{imdb_base_path}/movies-dataset",
        'imdb_links_path': f"{imdb_base_path}/movies-dataset/links.csv",
        'imdb_movies_metadata_path': f"{imdb_base_path}/movies-dataset/movies_metadata.csv",
        'imdb_output_full_path': f"{imdb_base_path}/movies-dataset/full_data.csv"
    })