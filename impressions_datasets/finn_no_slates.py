import os
from functools import cached_property
from typing import cast

import dask.array as da
import numba as nb
import numpy as np
from dask import delayed
from numpy.lib.npyio import NpzFile


@nb.njit
def compute_items_in_slate(
    arr_click: np.ndarray,
    arr_slates: np.ndarray,
    arr_time_steps: np.ndarray,
):
    num_rows = arr_click.shape[0]
    arr_item_in_slate = np.zeros_like(arr_click, dtype=np.bool_)

    for row_idx in np.arange(num_rows):
        for time_step in arr_time_steps:
            item_id = arr_click[row_idx, time_step]
            slate = arr_slates[row_idx, time_step, :]
            item_in_slate = item_id in slate

            arr_item_in_slate[row_idx, time_step] = item_in_slate

    return arr_item_in_slate


class DatasetFINNNoSlates:
    def __init__(self, dir_datasets: str):
        self.dir_dataset = os.path.join(
            dir_datasets,
            "FINN-no-slates",
            "",
        )
        self.file_data = os.path.join(
            self.dir_dataset,
            "data32.npz",
        )

        self.num_raw_data_points = 2_277_645
        self.num_time_steps = 20
        self.num_items_in_each_impression = 25
        self.num_data_points = (
            45_552_900  # MUST BE NUM_RAW_DATA_POINTS * NUM_TIME_STEPS
        )

    @cached_property
    def users(self) -> da.Array:
        with cast(NpzFile, np.load(file=self.file_data)) as dataset:
            return da.from_array(dataset["userId"])

    @cached_property
    def items(self) -> da.Array:
        with cast(NpzFile, np.load(file=self.file_data)) as dataset:
            return da.from_array(dataset["click"])

    @cached_property
    def slates(self) -> da.Array:
        with cast(NpzFile, np.load(file=self.file_data)) as dataset:
            return da.from_array(dataset["slate"])

    @cached_property
    def issue(self):
        with cast(NpzFile, np.load(file=self.file_data)) as dataset:
            arr_users = dataset["userId"]
            arr_click = dataset["click"]
            arr_slates = dataset["slate"]

            arr_time_steps = np.arange(20, dtype=np.int32)

            assert arr_users.shape[0] == self.num_raw_data_points
            assert arr_click.shape == (
                self.num_raw_data_points,
                self.num_time_steps,
            )
            assert arr_slates.shape == (
                self.num_raw_data_points,
                self.num_time_steps,
                self.num_items_in_each_impression,
            )

            arr_item_in_slate = compute_items_in_slate(
                arr_click=arr_click,
                arr_slates=arr_slates,
                arr_time_steps=arr_time_steps,
            )
            assert arr_item_in_slate.shape == (
                self.num_raw_data_points,
                self.num_time_steps,
            )

        return arr_item_in_slate, arr_users[0], arr_click[0, :], arr_slates[0, :, :]


class StatisticsFINNNoSlates:
    def __init__(self, dataset: DatasetFINNNoSlates, dir_statistics: str):
        self.dataset = dataset

        self.dir_statistics = os.path.join(
            dir_statistics,
            "FINN-no-slates",
            "",
        )

    @cached_property
    def num_users(self) -> int:
        arr_unique_users = da.unique(
            self.dataset.users,
            return_index=False,
            return_inverse=False,
            return_counts=False,
        )
        arr_unique_users.compute_chunk_sizes()

        return arr_unique_users.shape[0]

    @cached_property
    def num_items(self) -> int:
        arr_unique_items_interactions = da.unique(
            self.dataset.items,
            return_index=False,
            return_inverse=False,
            return_counts=False,
        )
        arr_unique_items_slates = da.unique(
            self.dataset.slates,
            return_index=False,
            return_inverse=False,
            return_counts=False,
        )

        arr_unique_items = da.unique(
            da.concatenate(
                [
                    arr_unique_items_interactions,
                    arr_unique_items_slates,
                ],
            ),
            return_index=False,
            return_inverse=False,
            return_counts=False,
        )
        arr_unique_items.compute_chunk_sizes()

        # Return the number of unique items minus 2 to remove the "non-clicked" item (1) and the "pad" item (0)
        return arr_unique_items.shape[0] - 2

    @cached_property
    def num_interactions(self) -> int:
        return (self.dataset.items >= 2).sum()

    @cached_property
    def num_interactions_non_clicked(self) -> int:
        return (self.dataset.items == 1).sum()

    @cached_property
    def num_impressions(self) -> int:
        # We use the click-idx column to tell where in the recommendation list the user clicked. What we do
        # is to set those items as -1 (internally, called interacted).
        # We need to remove all zeros in the recommendation list (they are padding items).
        # We need to remove all ones in the recommendation list (they are "no-click" items).
        # After this processing, our recommendation lists contain only impressed items.
        # Hence, We do a boolean mask of items greater than 1 and sum the boolean mask to count impressions.

        # The number of impressions is the number of items in recommendation lists
        # minus the number of pad items (item_id=0)
        # minus the number of non-clicked items (item_id=1)
        # minus the interacted items (dictated by "interaction_pos")
        num_recommended_items = self.dataset.slates.size
        num_pad_items = (self.dataset.slates == 0).sum()
        num_non_clicked_items = (self.dataset.slates == 1).sum()
        num_interacted_items = self.num_interactions

        return (
            num_recommended_items
            - num_pad_items
            - num_non_clicked_items
            - num_interacted_items
        )

    @cached_property
    def num_recommendations(self) -> int:
        # We use two ways to compute the number of recommendations.

        # One only uses the dataset and its description.
        # First, it removes items in the recommendation list with id=0 as they are "pad items", i.e., codes used to
        # fill the recommendation list.
        # Second, it removes items in the recommendation list with id=1 as it's a code for "non-clicks".
        num_recommended_items_dataset = (
            self.dataset.slates.size
            - (self.dataset.slates == 0).sum()
            - (self.dataset.slates == 1).sum()
        )

        # The second approach only uses information inside the dataset, i.e., it uses the `num_recommendations` columns
        # that tell us the number of recommended items to the user. However, there is a discrepancy between this number
        # and the number of items in the recommendations lists coming in the dataset.
        # num_recommended_items_reported = self.dataset.df["num_recommendations"].sum()

        return num_recommended_items_dataset  # , num_recommended_items_reported

    def statistics(self):
        statistics = delayed(
            {
                "dataset": "FINN.no Slates",
                "num_users": self.num_users,
                "num_items": self.num_items,
                "num_impressions": self.num_impressions,
                "num_interactions": self.num_interactions,
                # "num_non_clicked": self.num_interactions_non_clicked,
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
