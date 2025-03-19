import os
import pandas as pd

from src.loader.downloader import Downloader
from src.utils import defaults as d
from src.utils.enums import MovieLensType, MovieLensDataset
from config import settings

def define_schema_pandas(df: pd.DataFrame, ml_type: MovieLensType) -> pd.DataFrame:
    if ml_type == MovieLensType.RATINGS:
        df = df.rename(columns={
            'userId': d.idf_user,
            'movieId': d.idf_item,
            'rating': d.idf_rating,
            'timestamp': d.idf_timestamp
        })
    if ml_type == MovieLensType.MOVIES:
        df = df.rename(columns={
            'movieId': d.idf_item,
            'title': d.idf_title,
            'genres': d.idf_genres
        })
    return df


class Loader:
    def __init__(self,
                 download_folder=settings.get("DOWNLOAD_FOLDER", os.environ.get("DOWNLOAD_FOLDER")),
                 extract_folder=settings.get("DOWNLOAD_FOLDER", os.environ.get("DOWNLOAD_FOLDER"))):
        self.extract_folder = extract_folder
        self.download_folder = download_folder

    def _get_file_path(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> str:
        print(self.extract_folder)
        print(dataset.name)

        dataset_folder = os.path.join(self.extract_folder, dataset.name)
        file_name = ml_type.value
        if dataset == MovieLensDataset.ML_1M:
            file_name = file_name.replace('.csv', '.dat')

        return os.path.join(dataset_folder, file_name)

    def _ensure_dataset(self, dataset: MovieLensDataset):
        dataset_folder = os.path.join(self.extract_folder, dataset.name)
        if not os.path.exists(dataset_folder):
            print(f"Dataset {dataset.name} não encontrado em {dataset_folder}. Iniciando download e extração...")
            downloader = Downloader(download_folder=self.download_folder, extract_folder=self.extract_folder)
            downloader.download_and_extract_dataset(dataset)
        else:
            print(f"Dataset {dataset.name} já existe em {dataset_folder}.")

    def load_pandas(self, dataset: MovieLensDataset, ml_type: MovieLensType) -> pd.DataFrame:
        file_path = self._get_file_path(dataset, ml_type)
        if not os.path.exists(file_path):
            self._ensure_dataset(dataset)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado mesmo após o download.")

        if dataset == MovieLensDataset.ML_1M and ml_type == MovieLensType.RATINGS:
            df = pd.read_csv(file_path, sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"], encoding='ISO-8859-1')
        elif dataset == MovieLensDataset.ML_1M and ml_type == MovieLensType.MOVIES:
            df = pd.read_csv(file_path, sep="::", engine="python", names=["movieId", "title", "genres"], encoding='ISO-8859-1')
        else:
            df = pd.read_csv(file_path)
        return define_schema_pandas(df, ml_type)