import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

MOVIE_HEADERS = [
    "movieId", "title", "releaseDate", "videoReleaseDate", "IMDb URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
USER_HEADERS = ["userId", "age", "gender", "occupation", "zipCode"]
RATING_HEADERS = ["userId", "movieId", "rating", "timestamp"]


class KGMovieLens100K(InMemoryDataset):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

    def __init__(
        self,
        root: str = './data/kg/ml-100k',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        split: str = 'train',
        use_only_item_user: bool = False,
    ):
        self.split = split
        self.use_only_item_user = use_only_item_user
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0 if split=='train' else 1], data_cls=HeteroData)
        self.num_users, self.num_items = self.data['user'].num_nodes, self.data['movie'].num_nodes
        self.data = self.data.to_homogeneous()

    @property
    def raw_file_names(self) -> List[str]:
        return ['u.item', 'u.user', 'u1.base', 'u1.test']

    @property
    def processed_file_names(self) -> str:
        return [f'train_data_{self.use_only_item_user}.pt', 'test_data.pt']

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-100k')
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self):
        import pandas as pd

        data = HeteroData()

        # Process user data:
        df_user = pd.read_csv(
            self.raw_paths[1],
            sep='|',
            header=None,
            names=USER_HEADERS,
            index_col='userId',
            encoding='ISO-8859-1',
        )
        user_mapping = {idx: i for i, idx in enumerate(df_user.index)}
        age_mapping = {age: i for i, age in enumerate(df_user['age'].drop_duplicates().sort_values())}
        gender_mapping = {'M': 0, 'F': 1}
        zipcode_mapping = {zipcode: i for i, zipcode in enumerate(df_user['zipCode'].drop_duplicates())}
        occupation_mapping = {occupation: i for i, occupation in enumerate(df_user['occupation'].drop_duplicates())}

        # Process movie data:
        df_movie = pd.read_csv(
            self.raw_paths[0],
            sep='|',
            header=None,
            names=MOVIE_HEADERS,
            index_col='movieId',
            encoding='ISO-8859-1',
        )
        genre_columns = MOVIE_HEADERS[5:]

        def row_to_genres(row):
            genres = [genre.lower() for genre, value in row.items() if value == 1 and genre in genre_columns]
            return genres
        
        df_movie['genre'] = df_movie.apply(row_to_genres, axis=1)
        df_movie = df_movie.drop(columns=genre_columns)
        df_movie['year'] = df_movie['releaseDate'].str[-4:]
        movie_mapping = {idx: i for i, idx in enumerate(df_movie.index)}
        genre_mapping = {genre: i for i, genre in enumerate(df_movie['genre'].explode().drop_duplicates())}
        year_mapping = {year: i for i, year in enumerate(df_movie['year'].drop_duplicates().sort_values())}
        
        if self.split=='train':

            # Process rating data for training:
            df_rating = pd.read_csv(
                self.raw_paths[2],
                sep='\t',
                header=None,
                names=RATING_HEADERS,
            )

            # Add user movie connections
            src = [user_mapping[idx] for idx in df_rating['userId']]
            dst = [movie_mapping[idx] for idx in df_rating['movieId']]
            edge_index = torch.tensor([src, dst])
            data['user', 'rates', 'movie'].edge_index = edge_index

            data['user'].num_nodes = len(user_mapping)
            data['movie'].num_nodes = len(movie_mapping)

            if not self.use_only_item_user:
                # Add user gender connections
                src = [user_mapping[idx] for idx in df_user.index]
                dst = [gender_mapping[gender] for gender in df_user['gender']]
                data['user', 'is_gender', 'gender'].edge_index = torch.tensor([src, dst])
                data['gender'].num_nodes = 2

                # Add user occupation connections
                src = [user_mapping[idx] for idx in df_user.index]
                dst = [occupation_mapping[occupation] for occupation in df_user['occupation']]
                data['user', 'works_as', 'occupation'].edge_index = torch.tensor([src, dst])
                data['occupation'].num_nodes = len(occupation_mapping)

                # Add user zipcode connections
                src = [user_mapping[idx] for idx in df_user.index]
                dst = [zipcode_mapping[zipcode] for zipcode in df_user['zipCode']]
                data['user', 'lives_in', 'zipcode'].edge_index = torch.tensor([src, dst])
                data['zipcode'].num_nodes = len(zipcode_mapping)

                # Add user age connections
                src = [user_mapping[idx] for idx in df_user.index]
                dst = [age_mapping[age] for age in df_user['age']]
                data['user', 'has_years', 'age'].edge_index = torch.tensor([src, dst])
                data['age'].num_nodes = len(age_mapping)

                # Add movie genre connections
                src = [movie_mapping[idx] for idx in df_movie.explode('genre').index]
                dst = [genre_mapping[genre] for genre in df_movie['genre'].explode()]
                data['movie', 'is_type', 'genre'].edge_index = torch.tensor([src, dst])
                data['genre'].num_nodes = len(genre_mapping)

                # Add movie year connections
                src = [movie_mapping[idx] for idx in df_movie[pd.notnull(df_movie['year'])].index]
                dst = [year_mapping[year] for year in df_movie[pd.notnull(df_movie['year'])]['year'].explode()]
                data['movie', 'was_released', 'year'].edge_index = torch.tensor([src, dst])
                data['year'].num_nodes = len(year_mapping)

        else:
            # Process rating data for testing:
            df = pd.read_csv(
                self.raw_paths[3],
                sep='\t',
                header=None,
                names=RATING_HEADERS,)
            # Add user movie connections
            src = [user_mapping[idx] for idx in df['userId']]
            dst = [movie_mapping[idx] for idx in df['movieId']]
            edge_index = torch.tensor([src, dst])
            # only users in test df, but all movies
            data['user'].num_nodes = len(user_mapping)
            data['movie'].num_nodes = len(movie_mapping)
            data['user', 'rates', 'movie'].edge_index = edge_index

        self.save([data], f'{self.processed_paths[0 if self.split=="train" else 1]}')