import pandas as pd


def load_data_from_df(df: pd.DataFrame):
    user_num = df['user_id'].max()
    item_num = df['movie_id'].max()

    train_data = {}
    valid_data = {}
    test_data = {}

    # Ordena o DataFrame pelo timestamp e agrupa por usuário
    df_sorted = df.sort_values(by="timestamp")
    grouped = df_sorted.groupby("user_id")

    # Para cada usuário, obtém a lista de filmes em ordem cronológica
    for user, group in grouped:
        items = group['movie_id'].tolist()
        if len(items) < 3:
            train_data[user] = items
            valid_data[user] = []
            test_data[user] = []
        else:
            train_data[user] = items[:-2]
            valid_data[user] = [items[-2]]
            test_data[user] = [items[-1]]

    return train_data, valid_data, test_data, user_num, item_num
