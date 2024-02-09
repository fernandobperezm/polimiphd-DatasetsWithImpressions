import json
import os
from functools import cached_property

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed


class DatasetContentWiseImpressions:
    def __init__(self, dir_datasets: str):
        self.dir_dataset = os.path.join(
            dir_datasets,
            "ContentWiseImpressions",
            "CW10M",
            "",
        )
        self.dir_interactions = os.path.join(
            self.dir_dataset,
            "interactions",
            "",
        )
        self.dir_impressions_direct_link = os.path.join(
            self.dir_dataset,
            "impressions-direct-link",
            "",
        )
        self.dir_impressions_non_direct_link = os.path.join(
            self.dir_dataset,
            "impressions-non-direct-link",
            "",
        )
        self.file_metadata = os.path.join(
            self.dir_dataset,
            "metadata.json",
        )

        self.engine = "pyarrow"

    @cached_property
    def metadata(self) -> dict:
        with open(self.file_metadata, "r") as metadata_file:
            return json.load(
                metadata_file,
            )

    @cached_property
    def interactions(self) -> dd.DataFrame:
        return dd.read_parquet(
            path=self.dir_interactions,
            engine=self.engine,
        )

    @cached_property
    def impressions(self) -> dd.DataFrame:
        return dd.read_parquet(
            path=self.dir_impressions_direct_link,
            engine=self.engine,
        )

    @cached_property
    def impressions_non_direct_link(self) -> dd.DataFrame:
        return dd.read_parquet(
            path=self.dir_impressions_non_direct_link,
            engine=self.engine,
        )


class StatisticsContentWiseImpressions:
    def __init__(self, dataset: DatasetContentWiseImpressions, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "ContentWiseImpressions",
            "",
        )

    @cached_property
    def num_users(self) -> int:
        df_users_interactions: dd.Series = self.dataset.interactions["user_id"].unique()
        df_users_impressions: dd.Series = (
            self.dataset.impressions_non_direct_link.reset_index(drop=False)[
                "user_id"
            ].unique()
        )

        return dd.concat(
            dfs=[df_users_interactions, df_users_impressions],
            axis="index",
        ).nunique()

    @cached_property
    def num_items(self) -> int:
        return self.dataset.interactions["item_id"].unique()

    @cached_property
    def num_series(self) -> int:
        df_items_interactions: dd.Series = self.dataset.interactions[
            "series_id"
        ].unique()
        df_items_impressions: dd.Series = (
            self.dataset.impressions["recommended_series_list"]
            .explode()
            .rename("series_id")
            .unique()
        )
        df_items_impressions_non_direct_link: dd.Series = (
            self.dataset.impressions_non_direct_link["recommended_series_list"]
            .explode()
            .rename("series_id")
            .unique()
        )

        return dd.concat(
            dfs=[
                df_items_interactions,
                df_items_impressions,
                df_items_impressions_non_direct_link,
            ],
            axis="index",
        ).nunique()

    @cached_property
    def num_interactions(self) -> int:
        return self.dataset.interactions.shape[0]

    @cached_property
    def num_impressions(self) -> int:
        def row_by_row(df_row: dd.DataFrame):
            series_id = df_row["series_id"]
            list_recommendations = df_row["recommended_series_list"]

            if list_recommendations is None or len(list_recommendations) == 0:
                return 0

            num_recommendations = len(list_recommendations)
            num_interactions = np.isin(
                series_id,
                list_recommendations,
                assume_unique=False,
                invert=False,
            ).sum()

            return num_recommendations - num_interactions

        # This dataframe contains the following columns: user_id: int | recommendation_id: int >= 0 | series_id: list[int]
        # Where series_id is a list of unique interacted items within the same recommendation list.
        df_grouped_interactions: dd.DataFrame = (
            self.dataset.interactions[
                self.dataset.interactions["recommendation_id"] >= 0
            ]
            .groupby(by=["user_id", "recommendation_id"])["series_id"]
            .unique()
            .reset_index(drop=False)
        )

        # This dataframe is the same as before but has the recommendation list in it.
        df_series_with_impressions: dd.DataFrame = dd.merge(
            left=df_grouped_interactions,
            right=self.dataset.impressions[["recommended_series_list"]],
            left_index=False,
            right_index=True,
            left_on="recommendation_id",
            right_on=None,
            how="inner",
            suffixes=("", ""),
        )

        num_impressions_direct_link = df_series_with_impressions.apply(
            row_by_row,
            axis=1,
            meta=pd.Series(dtype=np.int32),
        ).sum()

        num_impressions_non_direct_link = self.dataset.impressions_non_direct_link[
            "recommendation_list_length"
        ].sum()

        return num_impressions_direct_link + num_impressions_non_direct_link

    @cached_property
    def num_recommendations(self) -> int:
        num_impressions = self.dataset.impressions["recommendation_list_length"].sum()
        num_impressions_non_direct_link = self.dataset.impressions_non_direct_link[
            "recommendation_list_length"
        ].sum()

        return num_impressions + num_impressions_non_direct_link

    def statistics(self) -> dict[str, object]:
        statistics = delayed(
            {
                "dataset": "ContentWise Impressions",
                "num_users": self.num_users,
                "num_items": self.num_series,
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
