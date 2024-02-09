import os
from functools import cached_property
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed


class DatasetAlimama:
    def __init__(self, dir_datasets: str):
        self.dir_dataset = os.path.join(
            dir_datasets,
            "Alimama",
            "",
        )
        self.file_raw_sample = os.path.join(
            self.dir_dataset,
            "raw_sample.csv",
        )

        # 33.46 - 25.52, 36.14 - 17.69
        self.dtypes_raw_sample = {
            "user": np.int32,
            "time_stamp": np.int64,
            "adgroup_id": np.int32,
            "pid": pd.StringDtype(),
            "nonclk": pd.BooleanDtype(),
            "clk": pd.BooleanDtype(),
        }

    @cached_property
    def raw_sample(self) -> dd.DataFrame:
        """
        Returns
        -------
        dd.DataFrame
        """
        return dd.read_table(
            urlpath=self.file_raw_sample,
            sep=",",
            dtype=self.dtypes_raw_sample,
            # blocksize="256MB",
        )


class StatisticsAlimama:
    def __init__(self, dataset: DatasetAlimama, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "Alimama",
            "",
        )
        self.df = self.dataset.raw_sample

    @cached_property
    def num_users(self) -> int:
        return self.df["user"].nunique()

    @cached_property
    def num_items(self) -> int:
        return self.df["adgroup_id"].nunique()

    @cached_property
    def num_interactions(self) -> int:
        return self.df["clk"].sum()

    @cached_property
    def num_impressions(self) -> int:
        return self.df["nonclk"].sum()

    @cached_property
    def num_recommendations(self) -> int:
        return self.df.shape[0]

    def statistics(self) -> dict[str, Any]:
        statistics = delayed(
            {
                "dataset": "Alimama",
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
