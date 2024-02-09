import os
from functools import cached_property
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed


class DatasetComboFashion:
    def __init__(self, dir_datasets: str):
        self.dir_dataset = os.path.join(
            dir_datasets,
            "Combo-Fashion",
            "",
        )
        self.file_in_shops = os.path.join(
            self.dir_dataset,
            "kdd2022_combo_item_public_sample_nouser_in.txt",
        )
        self.file_cross_shops = os.path.join(
            self.dir_dataset,
            "kdd2022_combo_item_public_sample_nouser_cross.txt",
        )

        self.dtypes = {
            "user_id": np.int32,
            "content_id": np.int32,
            "label": pd.BooleanDtype(),
            "server_time": np.int64,
            "day": np.int32,
            "content__item_id_1": np.int32,
            "content__item_id_2": np.int32,
            # The following are non-relevant but explicitly mentioned due to Dask having parsing errors
            "content__brand_id_1": np.float32,
            "content__brand_id_2": np.float32,
            "content__cate_id_1": np.float32,
            "content__cate_id_2": np.float32,
            "content__dense1": np.float32,
            "content__dense2": np.float32,
            "content__price_bin_1": np.float32,
            "content__price_bin_2": np.float32,
            "content__seller_id_1": np.float32,
            "content__seller_id_2": np.float32,
        }

    @cached_property
    def in_shops(self) -> dd.DataFrame:
        """
        Returns
        -------
        dd.DataFrame
        """
        return dd.read_csv(
            urlpath=self.file_in_shops,
            sep=";",
            dtype=self.dtypes,
        )

    @cached_property
    def cross_shops(self) -> dd.DataFrame:
        """

        Returns
        -------
        dd.DataFrame
        """
        return dd.read_csv(
            urlpath=self.file_cross_shops,
            sep=";",
            dtype=self.dtypes,
        )


class StatisticsComboFashionInShops:
    def __init__(self, dataset: DatasetComboFashion, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "Combo-Fashion-Cross-Shops",
            "",
        )
        self.df = self.dataset.in_shops

    @cached_property
    def num_users(self) -> int:
        return self.df["user_id"].nunique()

    @cached_property
    def num_items(self) -> int:
        return self.df["content_id"].nunique()

    @cached_property
    def num_interactions(self) -> int:
        return (self.df["label"] == 1).sum()

    @cached_property
    def num_impressions(self) -> int:
        return (self.df["label"] == 0).sum()

    @cached_property
    def num_recommendations(self) -> int:
        return self.df.shape[0]

    def statistics(self) -> dict[str, Any]:
        statistics = delayed(
            {
                "dataset": "Combo-Fashion In-Shop",
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


class StatisticsComboFashionCrossShops(StatisticsComboFashionInShops):
    def __init__(self, dataset: DatasetComboFashion, dir_statistics: str):
        super().__init__(
            dataset=dataset,
            dir_statistics=dir_statistics,
        )

        self.dir_statistics = os.path.join(
            dir_statistics,
            "Combo-Fashion-Cross-Shops",
            "",
        )
        self.df = self.dataset.cross_shops

    def statistics(self):
        statistics = super().statistics()
        statistics["dataset"] = "Combo-Fashion Cross-Shop"

        return statistics
