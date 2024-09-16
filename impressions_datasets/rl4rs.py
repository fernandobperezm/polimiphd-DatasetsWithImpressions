import os
from functools import cached_property

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed


class DatasetRL4RS:
    def __init__(self, dir_datasets: str):
        self.dir_dataset = os.path.join(
            dir_datasets,
            "rl4rs-dataset",
            "",
        )
        self.filepath_item_info = os.path.join(
            self.dir_dataset,
            "item_info.csv",
        )
        self.filepath_rl4rs_dataset_a_rl = os.path.join(
            self.dir_dataset,
            "rl4rs_dataset_a_rl.csv",
        )
        self.filepath_rl4rs_dataset_a_sl = os.path.join(
            self.dir_dataset,
            "rl4rs_dataset_a_sl.csv",
        )
        self.filepath_rl4rs_dataset_b_rl = os.path.join(
            self.dir_dataset,
            "rl4rs_dataset_b_rl.csv",
        )
        self.filepath_rl4rs_dataset_b_sl = os.path.join(
            self.dir_dataset,
            "rl4rs_dataset_b_sl.csv",
        )

        self.kwargs_rl4rs = {
            "sep": "@",
            "dtype": {
                "timestamp": np.int64,
                "session_id": np.int32,
                "sequence_id": np.int32,
                "exposed_items": pd.StringDtype(),
                "user_feedback": pd.StringDtype(),
                "user_seqfeature": pd.StringDtype(),
                "user_protrait": pd.StringDtype(),
                "item_feature": pd.StringDtype(),
                "behavior_policy_id": np.int32,
            },
            "encoding": "utf-8",
            "on_bad_lines": "error",
        }

    @cached_property
    def item_info(self) -> dd.DataFrame:
        return dd.read_csv(
            urlpath=self.filepath_item_info,
            sep=" ",
            encoding="utf-8",
            on_bad_lines="error",
            dtype={
                "item_id": np.int32,
                "item_vec": pd.StringDtype(),
                "price": np.float32,
                "location": np.int32,
                "special_item": np.int32,
            },
        )

    @cached_property
    def rl4rs_dataset_a_rl(self) -> dd.DataFrame:
        return dd.read_csv(
            urlpath=self.filepath_rl4rs_dataset_a_rl,
            **self.kwargs_rl4rs,
        )

    @cached_property
    def rl4rs_dataset_a_sl(self) -> dd.DataFrame:
        return dd.read_csv(
            urlpath=self.filepath_rl4rs_dataset_a_sl,
            **self.kwargs_rl4rs,
        )

    @cached_property
    def rl4rs_dataset_b_rl(self) -> dd.DataFrame:
        return dd.read_csv(
            urlpath=self.filepath_rl4rs_dataset_b_rl,
            **self.kwargs_rl4rs,
        )

    @cached_property
    def rl4rs_dataset_b_sl(self) -> dd.DataFrame:
        return dd.read_csv(
            urlpath=self.filepath_rl4rs_dataset_b_sl,
            **self.kwargs_rl4rs,
        )


class StatisticsRL4RS:
    def __init__(self, dataset: DatasetRL4RS, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "RL4RS",
            "",
        )

    @cached_property
    def num_users(self) -> int:
        df_users_a_rl: dd.Series = self.dataset.rl4rs_dataset_a_rl[
            "user_protrait"
        ].unique()
        df_users_b_rl: dd.Series = self.dataset.rl4rs_dataset_b_rl[
            "user_protrait"
        ].unique()

        df_users_a_sl: dd.Series = self.dataset.rl4rs_dataset_a_sl[
            "user_protrait"
        ].unique()
        df_users_b_sl: dd.Series = self.dataset.rl4rs_dataset_b_sl[
            "user_protrait"
        ].unique()

        return dd.concat(
            dfs=[
                df_users_a_rl,
                df_users_b_rl,
                df_users_a_sl,
                df_users_b_sl,
            ],
            axis="index",
        ).nunique()

    @cached_property
    def num_items(self) -> int:
        # The paper says all items are in this file.
        return self.dataset.item_info["item_id"].nunique()

        # df_info_unique_items = self.dataset.item_info["item_id"].unique()

        # df_unique_items_a_rl = pd.Series(
        #     self.dataset.rl4rs_dataset_a_rl["exposed_items"]
        #     .compute()
        #     .str.split(",")
        #     .explode()
        #     .unique()
        # )
        # df_unique_items_b_rl = pd.Series(
        #     self.dataset.rl4rs_dataset_b_rl["exposed_items"]
        #     .compute()
        #     .str.split(",")
        #     .explode()
        #     .unique()
        # )

        # df_unique_items_a_sl = pd.Series(
        #     self.dataset.rl4rs_dataset_a_sl["exposed_items"]
        #     .compute()
        #     .str.split(",")
        #     .explode()
        #     .unique()
        # )
        # df_unique_items_b_sl = pd.Series(
        #     self.dataset.rl4rs_dataset_b_sl["exposed_items"]
        #     .compute()
        #     .str.split(",")
        #     .explode()
        #     .unique()
        # )

        # return dd.concat(
        #     dfs=[
        #         df_info_unique_items,
        #         df_unique_items_a_rl,
        #         df_unique_items_b_rl,
        #         df_unique_items_a_sl,
        #         df_unique_items_b_sl,
        #     ],
        #     axis="index",
        # ).nunique()

    @cached_property
    def num_interactions(self) -> int:
        # The 'user_feedback' column holds arrays containing 1s for interactions or 0s for non-interacted impressions.
        # There is a digit for every impressed item in the array.
        # Thus, the number of interactions is just the sum of 1s.
        def count_interactions_in_array(str_user_feedback: str) -> int:
            return np.asarray(
                str_user_feedback.split(","),
                dtype=np.int32,
            ).sum()

        df_num_interactions_a_rl = (
            self.dataset.rl4rs_dataset_a_rl["user_feedback"]
            .apply(
                count_interactions_in_array,
                meta=("num_interactions", np.int32),
            )
            .sum()
        )
        df_num_interactions_a_sl = (
            self.dataset.rl4rs_dataset_a_sl["user_feedback"]
            .apply(
                count_interactions_in_array,
                meta=("num_interactions", np.int32),
            )
            .sum()
        )
        df_num_interactions_b_rl = (
            self.dataset.rl4rs_dataset_b_rl["user_feedback"]
            .apply(
                count_interactions_in_array,
                meta=("num_interactions", np.int32),
            )
            .sum()
        )
        df_num_interactions_b_sl = (
            self.dataset.rl4rs_dataset_b_sl["user_feedback"]
            .apply(
                count_interactions_in_array,
                meta=("num_interactions", np.int32),
            )
            .sum()
        )

        return (
            df_num_interactions_a_rl
            + df_num_interactions_a_sl
            + df_num_interactions_b_rl
            + df_num_interactions_b_sl
        )

    @cached_property
    def num_impressions(self) -> int:
        # The 'exposed_items' column holds arrays containing N item identifiers, thus for each array, there are N impressions.
        def count_impressions_in_array(str_user_feedback: str) -> int:
            return len(str_user_feedback.split(","))

        df_num_impressions_a_rl = (
            self.dataset.rl4rs_dataset_a_rl["user_feedback"]
            .apply(
                count_impressions_in_array,
                meta=("num_impressions", np.int32),
            )
            .sum()
        )
        df_num_impressions_a_sl = (
            self.dataset.rl4rs_dataset_a_sl["user_feedback"]
            .apply(
                count_impressions_in_array,
                meta=("num_impressions", np.int32),
            )
            .sum()
        )
        df_num_impressions_b_rl = (
            self.dataset.rl4rs_dataset_b_rl["user_feedback"]
            .apply(
                count_impressions_in_array,
                meta=("num_impressions", np.int32),
            )
            .sum()
        )
        df_num_impressions_b_sl = (
            self.dataset.rl4rs_dataset_b_sl["user_feedback"]
            .apply(
                count_impressions_in_array,
                meta=("num_impressions", np.int32),
            )
            .sum()
        )

        return (
            df_num_impressions_a_rl
            + df_num_impressions_a_sl
            + df_num_impressions_b_rl
            + df_num_impressions_b_sl
        )

    def statistics(self) -> dict[str, object]:
        statistics = delayed(
            {
                "dataset": "RL4RS",
                "num_users": self.num_users,
                "num_items": self.num_items,
                "num_impressions": self.num_impressions,
                "num_interactions": self.num_interactions,
                "quota_interactions_impressions": (
                    self.num_interactions / self.num_impressions
                ),
                "quota_impressions_interactions": (
                    self.num_impressions / self.num_interactions
                ),
            }
        )

        return statistics.compute()
