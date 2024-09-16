import glob
import gzip
import os
from functools import cached_property, reduce
from typing import Any
from tqdm import tqdm

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import delayed


def _read_timestamp(line_contents: list[str], idx: int, arr_data: np.ndarray):
    timestamp = int(line_contents[0])
    arr_data[idx] = timestamp


def _read_article(line_contents: list[str], idx: int, arr_data: np.ndarray):
    article_id = line_contents[1]
    arr_data[idx] = article_id


def _read_label(line_contents: list[str], idx: int, arr_data: np.ndarray):
    arr_data[idx] = line_contents[2] == "1"


def _read_user_features(
    line_contents: list[str], idx: int, arr_data: np.ndarray
) -> int:
    list_features: list[str] = []
    idx_feat = 4
    for idx_feat, feat in enumerate(line_contents[4:], start=4):
        if feat.startswith("|"):
            break

        list_features.append(feat)

    arr_data[idx] = "-".join(np.sort(list_features))

    return idx_feat


def _read_candidate_items(
    line_contents: list[str], idx: int, idx_items_start: int, arr_data: np.ndarray
):
    # candidate_item_ids: list[str] = []
    # for item_str in line_contents[idx_items_start:]:
    #     candidate_item_ids.append(item_str.replace("|", "").replace("\n", ""))
    # arr_data[idx] = candidate_item_ids

    arr_data[idx] = ",".join(
        map(
            lambda item_str: item_str.replace("|", "").replace("\n", ""),
            line_contents[idx_items_start:],
        )
    )


def read_file(filepath: str) -> dd.DataFrame:
    # This obtains the filepath without the .gz extension.
    dir_parquet_file = os.path.splitext(filepath)[0]
    if os.path.exists(dir_parquet_file) and len(os.listdir(dir_parquet_file)) > 0:
        return dd.read_parquet(
            path=dir_parquet_file,
            engine="pyarrow",
        )

    with gzip.open(filepath, "rt") as f:
        num_lines = sum(1 for _ in f)

        list_timestamps: np.ndarray = np.empty(
            shape=(num_lines,),
            dtype=np.int64,
        )
        list_displayed_article_ids: np.ndarray = np.empty(
            shape=(num_lines,),
            dtype="object",
        )
        list_user_clicks: np.ndarray = np.empty(
            shape=(num_lines,),
            dtype=np.bool_,
        )
        list_user_features: np.ndarray = np.empty(
            shape=(num_lines,),
            dtype="object",
        )
        list_pool_articles_ids: np.ndarray = np.empty(
            shape=(num_lines,),
            dtype="object",
        )

        f.seek(0)

        for idx, line in enumerate(f):
            line_contents: list[str] = line.split(" ")

            _read_timestamp(
                line_contents=line_contents,
                idx=idx,
                arr_data=list_timestamps,
            )
            _read_article(
                line_contents=line_contents,
                idx=idx,
                arr_data=list_displayed_article_ids,
            )
            _read_label(
                line_contents=line_contents,
                idx=idx,
                arr_data=list_user_clicks,
            )

            # The index 3 is not important as it is constant ('|user') and denotes the start of user features.
            # user_delimiter = line[3]

            idx_items_start = _read_user_features(
                line_contents=line_contents,
                idx=idx,
                arr_data=list_user_features,
            )

            # _read_candidate_items(
            #     line_contents=line_contents,
            #     idx=idx,
            #     idx_items_start=idx_items_start,
            #     arr_data=list_pool_articles_ids,
            # )

        df = dd.from_pandas(
            data=pd.DataFrame.from_dict(
                data={
                    "timestamp": list_timestamps,
                    "displayed_article_id": list_displayed_article_ids,
                    "user_click": list_user_clicks,
                    "user_features": list_user_features,
                    # "pool_articles": list_pool_articles_ids,
                },
                orient="columns",
            ).astype(
                dtype={
                    "timestamp": np.int64,
                    "displayed_article_id": pd.StringDtype(),
                    "user_click": pd.BooleanDtype(),
                    "user_features": pd.StringDtype(),
                    # "pool_articles": pd.StringDtype(),
                },
            ),
            npartitions=1,
        )

        df.to_parquet(
            path=dir_parquet_file,
            engine="pyarrow",
        )

        return df


class DatasetYahooR6:
    def __init__(self, dir_datasets: str):
        self.dir_dataset_r6a = os.path.join(
            dir_datasets,
            "Yahoo-R6A",
            "",
        )
        self.dir_dataset_r6b = os.path.join(
            dir_datasets,
            "Yahoo-R6B",
            "",
        )

        self.files_dataset_r6a = os.path.join(
            self.dir_dataset_r6a,
            "ydata-fp-td-clicks-v1_0.*.gz",
        )
        self.files_dataset_r6b = os.path.join(
            self.dir_dataset_r6b,
            "ydata-fp-td-clicks-v2_0.*.gz",
        )

    @cached_property
    def r6a(self) -> list[dd.DataFrame]:
        files_to_read = glob.glob(self.files_dataset_r6a)
        return [read_file(fp) for fp in tqdm(files_to_read)]

    @cached_property
    def r6b(self) -> list[dd.DataFrame]:
        files_to_read = glob.glob(self.files_dataset_r6b)
        return [read_file(fp) for fp in tqdm(files_to_read)]


class StatisticsYahooR6A:
    def __init__(self, dataset: DatasetYahooR6, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "Yahoo-R6A",
            "",
        )
        self.dfs = self.dataset.r6a

    @cached_property
    def num_users(self) -> int:
        return dd.concat(
            dfs=[df["user_features"].unique() for df in self.dfs],
            axis="index",
        ).nunique()

    @cached_property
    def num_items(self) -> int:
        displayed_items: list[dd.Series] = [
            df["displayed_article_id"].unique() for df in self.dfs
        ]
        # pool_items: list[dd.Series] = [
        #     df["pool_articles"].str.split().explode().unique() for df in self.dfs
        # ]

        return dd.concat(
            dfs=(
                displayed_items
                # + pool_items
            ),
            axis="index",
        ).nunique()

    @cached_property
    def num_interactions(self) -> int:
        num_interactions_day = reduce(
            lambda acc, num_int: acc + num_int,
            map(
                lambda df: df[df["user_click"]].shape[0],
                self.dfs,
            ),
            0,
        )

        return num_interactions_day

    @cached_property
    def num_impressions(self) -> int:
        num_impressions_day = reduce(
            lambda acc, num_imp: acc + num_imp,
            map(
                lambda df: df[~df["user_click"]].shape[0],
                self.dfs,
            ),
            0,
        )

        return num_impressions_day

    @cached_property
    def num_recommendations(self) -> int:
        num_recommendations_day = reduce(
            lambda acc, num_rec: acc + num_rec,
            map(
                lambda df: df.shape[0],
                self.dfs,
            ),
            0,
        )

        return num_recommendations_day

    def statistics(self) -> dict[str, Any]:
        statistics = delayed(
            {
                "dataset": "Yahoo! R6A",
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


class StatisticsYahooR6B(StatisticsYahooR6A):
    def __init__(self, dataset: DatasetYahooR6, dir_statistics: str):
        super().__init__(
            dataset=dataset,
            dir_statistics=dir_statistics,
        )

        self.dir_statistics = os.path.join(
            dir_statistics,
            "Yahoo-R6B",
            "",
        )
        self.dfs = self.dataset.r6b

    def statistics(self) -> dict[str, Any]:
        statistics = super().statistics()
        statistics["dataset"] = "Yahoo! R6B"

        return statistics
