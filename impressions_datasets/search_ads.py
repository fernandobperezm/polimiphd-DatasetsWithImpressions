import os
from functools import cached_property

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed


class DatasetSearchAds:
    def __init__(self, dir_datasets: str):
        self.dir_dataset = os.path.join(
            dir_datasets,
            "SearchAds",
            "",
        )
        self.file_train = os.path.join(
            self.dir_dataset,
            "training.txt",
        )
        self.file_test = os.path.join(
            self.dir_dataset,
            "test.txt",
        )
        self.file_solution = os.path.join(
            self.dir_dataset,
            "KDD_Track2_solution.csv",
        )
        self.columns_train = [
            "num_interactions",
            "num_impressions",
            "display_url",
            "ad_id",
            "advertiser_id",
            "session_depth",
            "ad_position",
            "query_id",
            "keyword_id",
            "title_id",
            "description_id",
            "user_id",
        ]
        self.columns_test = self.columns_train[2:]
        self.columns_solution = self.columns_train[:2] + ["Indicator"]
        self.dtypes = {
            "num_interactions": np.int32,
            "num_impressions": np.int32,
            "display_url": np.int64,
            "ad_id": np.int32,
            "advertiser_id": np.int32,
            "depth": np.int32,
            "position": np.int32,
            "query_id": np.int32,
            "keyword_id": np.int32,
            "title_id": np.int32,
            "description_id": np.int32,
            "user_id": np.int32,
        }

    @cached_property
    def train(self) -> dd.DataFrame:
        """Train split of the Search Ads dataset.

        Each row in this dataset refers to an aggregation (group-by operation) of the following columns: `user_id`,
        `ad_id`, `query_id`, `depth`, and `position`. Each row also contains the number of interactions and impressions
        (named `num_interactions` and `num_impressions`, respectively) that occurred within the same aggregation, i.e.,
        same user, ad, query, depth, and position.

        Returns
        -------
        dd.DataFrame
        """
        return dd.read_csv(
            urlpath=self.file_train,
            sep="\t",
            header=None,
            names=self.columns_train,
            dtype=self.dtypes,
        )

    @cached_property
    def test(self) -> dd.DataFrame:
        """Test split of the Search Ads dataset.

        Each row in this dataset refers to an aggregation (group-by operation) of the following columns: `user_id`,
        `ad_id`, `query_id`, `depth`, and `position`. The test set does not contain the number of interactions nor
        impressions.

        Returns
        -------
        dd.DataFrame
        """
        return dd.read_csv(
            urlpath=self.file_test,
            sep="\t",
            header=None,
            names=self.columns_test,
            dtype=self.dtypes,
        )

    @cached_property
    def solution(self) -> dd.DataFrame:
        """Test split of the Search Ads dataset.

        Each row in this dataset refers to an aggregation (group-by operation) of the following columns: `user_id`,
        `ad_id`, `query_id`, `depth`, and `position`. The test set does not contain the number of interactions nor
        impressions.

        Returns
        -------
        dd.DataFrame
        """
        return dd.read_csv(
            urlpath=self.file_solution,
            sep=",",
            header=0,
            names=self.columns_solution,
            dtype={
                "num_interactions": np.int32,
                "num_impressions": np.int32,
                "Indicator": pd.CategoricalDtype(
                    categories=["Public", "Private"],
                    ordered=False,
                ),
            },
        )


class StatisticsSearchAds:
    def __init__(self, dataset: DatasetSearchAds, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "SearchAds",
            "",
        )

    @cached_property
    def num_users(self) -> int:
        df_users_train: dd.Series = self.dataset.train["user_id"].unique()
        df_users_test: dd.Series = self.dataset.test["user_id"].unique()

        return dd.concat(
            dfs=[df_users_train, df_users_test],
            axis="index",
        ).nunique()

    @cached_property
    def num_items(self) -> int:
        df_items_train: dd.Series = self.dataset.train["ad_id"].unique()
        df_items_test: dd.Series = self.dataset.test["ad_id"].unique()

        return dd.concat(
            dfs=[df_items_train, df_items_test],
            axis="index",
        ).nunique()

    @cached_property
    def num_interactions(self) -> int:
        # In this dataset, the num_interactions and num_impressions are given in a different dataframe (i.e., not the
        # `test` dataframe). That's why we use the `solution` dataframe instead of the `test` one.
        return (
            self.dataset.train["num_interactions"].sum()
            + self.dataset.solution["num_interactions"].sum()
        )

    @cached_property
    def num_impressions(self) -> int:
        # In this dataset, the num_interactions and num_impressions are given in a different dataframe (i.e., not the
        # `test` dataframe).
        return (
            self.dataset.train["num_impressions"]
            - self.dataset.train["num_interactions"]
        ).sum() + (
            self.dataset.solution["num_impressions"]
            - self.dataset.solution["num_interactions"]
        ).sum()

    @cached_property
    def num_recommendations(self) -> int:
        # In this dataset, the num_interactions and num_impressions are given in a different dataframe (i.e., not the
        # `test` dataframe).
        return (
            self.dataset.train["num_impressions"].sum()
            + self.dataset.solution["num_impressions"].sum()
        )

    def statistics(self):
        statistics = delayed(
            {
                "dataset": "SearchAds",
                "num_users": self.num_users,
                "num_items": self.num_items,
                "num_impressions": self.num_impressions,
                "num_interactions": self.num_interactions,
                "num_recommendations": self.num_recommendations,
                "quota_interactions_impressions": (
                    self.num_interactions / self.num_impressions
                ),
                "quota_impressions_interactions": (
                    self.num_impressions / self.num_interactions
                ),
            }
        )

        return statistics.compute()
