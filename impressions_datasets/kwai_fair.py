import os
from functools import cached_property
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed


class DatasetKwaiFair:
    def __init__(self, dir_datasets: str):
        self.dir_dataset = os.path.join(
            dir_datasets,
            "Kwai_Fair",
            "",
        )
        self.files_random_exposure = os.path.join(
            self.dir_dataset,
            "random_exposure",
            "day*_random.csv",
        )
        self.files_system_impression = os.path.join(
            self.dir_dataset,
            "system_impression",
            "day*_system.csv",
        )
        self.dtypes = {
            "photo_id": pd.StringDtype(),
            "user_id": pd.StringDtype(),
            "expose_time": np.int32,
            "play_time": np.float64,
            "is_click": pd.BooleanDtype(),
            "is_like": pd.BooleanDtype(),
            "is_comment": pd.BooleanDtype(),
            "is_follow": pd.BooleanDtype(),
            "is_forward": pd.BooleanDtype(),
            "is_dislike": pd.BooleanDtype(),
        }

    @cached_property
    def random_exposure(self) -> dd.DataFrame:
        """
        Returns
        -------
        dd.DataFrame
        """
        return dd.read_csv(
            urlpath=self.files_random_exposure,
            sep=",",
            dtype=self.dtypes,
        )

    @cached_property
    def system_exposure(self) -> dd.DataFrame:
        """

        Returns
        -------
        dd.DataFrame
        """
        return dd.read_csv(
            urlpath=self.files_system_impression,
            sep=",",
            dtype=self.dtypes,
        )


class StatisticsKwaiFairExperiment:
    def __init__(self, dataset: DatasetKwaiFair, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "Kwai-Fair-Experiment",
            "",
        )
        self.df = self.dataset.random_exposure

    @cached_property
    def num_users(self) -> int:
        return self.df["user_id"].nunique()

    @cached_property
    def num_items(self) -> int:
        return self.df["photo_id"].nunique()

    @cached_property
    def num_interactions(self) -> int:
        df_interactions = self.df[
            self.df["is_click"]
            | self.df["is_like"]
            | self.df["is_comment"]
            | self.df["is_follow"]
            | self.df["is_forward"]
            | self.df["is_dislike"]
        ]

        return df_interactions.shape[0]

    @cached_property
    def num_impressions(self) -> int:
        df_impressions = self.df[
            ~(
                self.df["is_click"]
                | self.df["is_like"]
                | self.df["is_comment"]
                | self.df["is_follow"]
                | self.df["is_forward"]
                | self.df["is_dislike"]
            )
        ]

        return df_impressions.shape[0]

    @cached_property
    def num_recommendations(self) -> int:
        return self.df.shape[0]

    def statistics(self) -> dict[str, Any]:
        statistics = delayed(
            {
                "dataset": "Kwai Fair Experiment",
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


class StatisticsKwaiFairSystem(StatisticsKwaiFairExperiment):
    def __init__(self, dataset: DatasetKwaiFair, dir_statistics: str):
        super().__init__(
            dataset=dataset,
            dir_statistics=dir_statistics,
        )

        self.dir_statistics = os.path.join(
            dir_statistics,
            "Kwai-Fair-System",
            "",
        )
        self.df = self.dataset.system_exposure

    def statistics(self):
        statistics = super().statistics()
        statistics["dataset"] = "Kwai Fair System"

        return statistics
