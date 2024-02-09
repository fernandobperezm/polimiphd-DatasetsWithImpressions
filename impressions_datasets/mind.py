import os
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed


def convert_impressions_str_to_array(
    impressions: Optional[str],
) -> list[str]:
    """Converts a string of recommended items to an array

    This function expects `impressions` to be a string as follows:
    - if None or NA, then this method returns [].
    - else, "NXXXX-Y NZZZZZ-Y NWWW-Y", i.e., a white-space separated string,
      where each item begins with an N followed by several numbers (the item id),
      then a dash character (-), then either 0 or 1. 1 Means the user interacted
      with this item, 0 otherwise.

    Notes
    -----
    Do not try to numba.jit decorate this function, given that numba does not have optimizations
    for the :py:mod:`re` module. Trying to decorate this function will cause it to fail on runtime.

    Returns
    -------
    list[str]
        a list containing the interacted item ids, if any, in the format "NXXXX". If the `impressions` string
        is None, empty, or non-interacted impression, then an empty list is returned.
        Else, a list containing the ids is returned.
    """
    if impressions is None or pd.isna(impressions) or impressions == "":
        return []

    return impressions.replace("-0", "").replace("-1", "").split(" ")


def extract_num_interactions_from_history_str(
    impressions: Optional[str],
) -> int:
    """Extract interacted items from history.

    This function expects `impressions` to be a string as follows:
    - if None or NA, then this method returns [].
    - else, "NXXXX-Y NZZZZZ-Y NWWW-Y", i.e., a white-space separated string,
      where each item begins with an N followed by several numbers (the item id),
      then a dash character (-), then either 0 or 1. 1 Means the user interacted
      with this item, 0 otherwise.

    Notes
    -----
    Do not try to numba.jit decorate this function, given that numba does not have optimizations
    for the :py:mod:`re` module. Trying to decorate this function will cause it to fail on runtime.

    Returns
    -------
    list[str]
        a list containing the interacted item ids, if any, in the format "NXXXX". If the `impressions` string
        is None, empty, or non-interacted impression, then an empty list is returned.
        Else, a list containing the ids is returned.
    """
    if impressions is None or pd.isna(impressions) or impressions == "":
        return 0

    return len(impressions.split(" "))


def extract_num_interactions_from_impressions_str(
    impressions: Optional[str],
) -> int:
    """Extract interacted items from impressions.

    This function expects `impressions` to be a string as follows:
    - if None or NA, then this method returns [].
    - else, "NXXXX-Y NZZZZZ-Y NWWW-Y", i.e., a white-space separated string,
      where each item begins with an N followed by several numbers (the item id),
      then a dash character (-), then either 0 or 1. 1 Means the user interacted
      with this item, 0 otherwise.

    Notes
    -----
    Do not try to numba.jit decorate this function, given that numba does not have optimizations
    for the :py:mod:`re` module. Trying to decorate this function will cause it to fail on runtime.

    Returns
    -------
    list[str]
        a list containing the interacted item ids, if any, in the format "NXXXX". If the `impressions` string
        is None, empty, or non-interacted impression, then an empty list is returned.
        Else, a list containing the ids is returned.
    """
    if impressions is None or pd.isna(impressions) or impressions == "":
        return 0

    return len(
        [
            recommendation.replace("-1", "")
            for recommendation in impressions.split(" ")
            if recommendation.endswith("-1")
        ]
    )


def extract_num_impressions_from_impressions_str(
    impressions: Optional[str],
) -> int:
    """Extract number of impressed items from the recommendations string.

    This function expects `impressions` to be a string as follows:
    - if None or NA, then this method returns [].
    - else, "NXXXX-Y NZZZZZ-Y NWWW-Y", i.e., a white-space separated string,
      where each item begins with an N followed by several numbers (the item id),
      then a dash character (-), then either 0 or 1. 1 Means the user interacted
      with this item, 0 otherwise.

    Notes
    -----
    Do not try to numba.jit decorate this function, given that numba does not have optimizations
    for the :py:mod:`re` module. Trying to decorate this function will cause it to fail on runtime.

    Returns
    -------
    list[str]
        a list containing the interacted item ids, if any, in the format "NXXXX". If the `impressions` string
        is None, empty, or non-interacted impression, then an empty list is returned.
        Else, a list containing the ids is returned.
    """
    if impressions is None or pd.isna(impressions) or impressions == "":
        return 0

    return len(
        [
            recommendation.replace("-0", "")
            for recommendation in impressions.split(" ")
            if recommendation.endswith("-0")
        ]
    )


class DatasetMINDLarge:
    def __init__(self, dir_datasets: str):
        self.dir_dataset = os.path.join(
            dir_datasets,
            "MIND-LARGE",
            "",
        )
        self.file_train = os.path.join(
            self.dir_dataset,
            "train",
            "behaviors.tsv",
        )
        self.file_validation = os.path.join(
            self.dir_dataset,
            "dev",
            "behaviors.tsv",
        )
        self.file_test = os.path.join(
            self.dir_dataset,
            "test",
            "behaviors.tsv",
        )

        self.names = [
            "impression_id",
            "user_id",
            "str_timestamp",
            "str_history",
            "str_impressions",
        ]
        self.dtypes = {
            "impression_id": np.int32,
            "user_id": pd.StringDtype(),
            "str_timestamp": pd.StringDtype(),
            "str_history": pd.StringDtype(),  # either 'object' or pd.StringDtype() produce the same column.
            "str_impressions": pd.StringDtype(),  # either 'object' or pd.StringDtype() produce the same column.
        }

        self.engine = "pyarrow"

    @cached_property
    def train(self) -> dd.DataFrame:
        return dd.read_csv(
            self.file_train,
            sep="\t",
            header=None,
            names=self.names,
            dtype=self.dtypes,
        )

    @cached_property
    def dev(self) -> dd.DataFrame:
        return dd.read_csv(
            self.file_validation,
            sep="\t",
            header=None,
            names=self.names,
            dtype=self.dtypes,
        )

    @cached_property
    def test(self) -> dd.DataFrame:
        return dd.read_csv(
            self.file_test,
            sep="\t",
            header=None,
            names=self.names,
            dtype=self.dtypes,
        )


class StatisticsMINDLarge:
    def __init__(self, dataset: DatasetMINDLarge, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "MIND-LARGE",
            "",
        )

    @cached_property
    def num_users(self) -> int:
        return dd.concat(
            dfs=[
                self.dataset.train["user_id"],
                self.dataset.dev["user_id"],
                self.dataset.test["user_id"],
            ],
            axis="index",
        ).nunique()

    @cached_property
    def num_items(self) -> int:
        df_items_history: dd.Series = (
            dd.concat(
                dfs=[
                    self.dataset.train["str_history"],
                    self.dataset.dev["str_history"],
                    self.dataset.test["str_history"],
                ],
                axis="index",
            )
            .map_partitions(
                func=lambda df: df.apply(
                    func=convert_impressions_str_to_array,
                ),
            )
            .explode()
            .unique()
        )

        df_items_impressions: dd.Series = (
            dd.concat(
                dfs=[
                    self.dataset.train["str_impressions"],
                    self.dataset.dev["str_impressions"],
                    self.dataset.test["str_impressions"],
                ],
                axis="index",
            )
            .map_partitions(
                func=lambda df: df.apply(
                    func=convert_impressions_str_to_array,
                ),
            )
            .explode()
            .unique()
        )

        return dd.concat(
            dfs=[df_items_history, df_items_impressions],
            axis="index",
        ).nunique()

    @cached_property
    def num_interactions(self) -> int:
        # The `test` split does not contain an indicator of impression or interaction, hence we do not use them to
        # compute the number of interactions (although we use that split to compute the number of recommendations)
        df_train_dev: dd.DataFrame = dd.concat(
            dfs=[
                self.dataset.train[["user_id", "str_history", "str_impressions"]],
                self.dataset.dev[["user_id", "str_history", "str_impressions"]],
            ],
            axis="index",
        )

        df_num_interactions_from_history: dd.DataFrame = (
            df_train_dev[["user_id", "str_history"]]
            .drop_duplicates(
                subset="user_id",
                keep="last",
            )["str_history"]
            .map_partitions(
                func=lambda df: df.apply(extract_num_interactions_from_history_str)
            )
        )

        df_num_interactions_from_recommendations: dd.DataFrame = df_train_dev[
            "str_impressions"
        ].map_partitions(
            func=lambda df: df.apply(extract_num_interactions_from_impressions_str)
        )

        return (
            df_num_interactions_from_history.sum()
            + df_num_interactions_from_recommendations.sum()
        )

    @cached_property
    def num_impressions(self) -> int:
        # The `test` split does not contain an indicator of impression or interaction, hence we do not use them to
        # compute the number of impressions (although we use that split to compute the number of recommendations)
        df_num_impressions = dd.concat(
            dfs=[
                self.dataset.train["str_impressions"],
                self.dataset.dev["str_impressions"],
            ],
            axis="index",
        ).map_partitions(
            func=lambda df: df.apply(extract_num_impressions_from_impressions_str)
        )

        return df_num_impressions.sum()

    @cached_property
    def num_recommendations(self) -> int:
        # The `test` split does not contain an indicator of impression or interaction, however, we use this split when
        # computing the number of recommendations as it is part of the dataset.
        df_num_impressions = dd.concat(
            dfs=[
                self.dataset.train["str_impressions"],
                self.dataset.dev["str_impressions"],
                self.dataset.test["str_impressions"],
            ],
            axis="index",
        ).map_partitions(
            func=lambda df: df.apply(
                lambda impression: (
                    0
                    if impression is None or pd.isna(impression) or impression == ""
                    else len(impression)
                )
            )
        )

        return df_num_impressions.sum()

    def statistics(self):
        statistics = delayed(
            {
                "dataset": "MIND",
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
