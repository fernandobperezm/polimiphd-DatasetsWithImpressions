import os
from functools import cached_property
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed


class DatasetPANDOR:
    def __init__(self, dir_datasets: str):
        self.dir_dataset = os.path.join(
            dir_datasets,
            "PANDOR",
            "",
        )
        self.file_data = os.path.join(
            self.dir_dataset,
            "fullData.anonymous",
        )

        self.dtypes = {
            # Dtypes with file first line.
            "source": pd.StringDtype(),  # "bb405212c420d3ef17516a69afcd997c145fc149f7ff0b302ae4ce5443901da4"
            "offerId": pd.StringDtype(),  # ";28292270338"
            "pageViewId": pd.StringDtype(),  # ";POM:85691523033533995"
            "offerViewId": pd.StringDtype(),  # ";POM:85691523033533995_28292270338"
            "utcDate": np.int64,  # ";5826113"
            "keywords": pd.StringDtype(),  # ";WrappedArray(home)"
            "wasClicked": pd.BooleanDtype(),  # ";false"
            "offerViewCountPerPageView": np.int64,  # ";1"
            "clickCountPerPageView": np.int64,  # ";0"
            "userId": pd.StringDtype(),  # ";bb73a1cd73ad760266bd0cd2ebb0fbd30e6eba23f7206421339ea29b6b459779"
            "productLemmas": "object",  # ";null"
            "productFeatures": "object",  # ";null"
            "url": pd.StringDtype(),  # ";0d4b103ae3ed2801d55cd4b2a06fde4ea75345ef9ad473ca63bc81a8580d91ef"
            "pageLemmas": "object",  # ";null"
            "pageFeatures": "object",  # ";null"
        }

    @cached_property
    def data(self) -> dd.DataFrame:
        """
        Returns
        -------
        dd.DataFrame
        """
        return dd.read_table(
            urlpath=self.file_data,
            sep=";",
            dtype=self.dtypes,
            blocksize="256MB",
        )


class StatisticsPANDOR:
    def __init__(self, dataset: DatasetPANDOR, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "PANDOR",
            "",
        )
        self.df = self.dataset.data

    @cached_property
    def num_users(self) -> int:
        return self.df["userId"].nunique()

    @cached_property
    def num_items(self) -> int:
        return self.df["offerId"].nunique()

    @cached_property
    def num_interactions(self) -> int:
        return self.df[self.df["wasClicked"]].shape[0]

    @cached_property
    def num_impressions(self) -> int:
        return self.df[~self.df["wasClicked"]].shape[0]

    @cached_property
    def num_recommendations(self) -> int:
        return self.df.shape[0]

    def statistics(self) -> dict[str, Any]:
        statistics = delayed(
            {
                "dataset": "PANDOR",
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
