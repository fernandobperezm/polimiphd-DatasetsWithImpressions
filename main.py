import pandas as pd
import os
import time
import traceback

from dask.distributed import Client, LocalCluster

from impressions_datasets.ali_ccp import DatasetAliCCP, StatisticsAliCCP
from impressions_datasets.alimama import DatasetAlimama, StatisticsAlimama
from impressions_datasets.finn_no_slates import (
    DatasetFINNNoSlates,
    StatisticsFINNNoSlates,
)
from impressions_datasets.mind import StatisticsMINDLarge, DatasetMINDLarge
from impressions_datasets.combo_fashion import (
    StatisticsComboFashionInShops,
    DatasetComboFashion,
    StatisticsComboFashionCrossShops,
)
from impressions_datasets.contentwise_impressions import (
    DatasetContentWiseImpressions,
    StatisticsContentWiseImpressions,
)
from impressions_datasets.kwai_fair import (
    DatasetKwaiFair,
    StatisticsKwaiFairExperiment,
    StatisticsKwaiFairSystem,
)
from impressions_datasets.pandor import DatasetPANDOR, StatisticsPANDOR
from impressions_datasets.search_ads import DatasetSearchAds, StatisticsSearchAds
from impressions_datasets.yahoo_r6_front_page import (
    StatisticsYahooR6B,
    DatasetYahooR6,
    StatisticsYahooR6A,
)
from impressions_datasets.rl4rs import (
    StatisticsRL4RS,
    DatasetRL4RS,
)

if __name__ == "__main__":
    dir_datasets = os.path.join(
        os.getcwd(),
        "data",
        "",
    )
    dir_statistics = os.path.join(
        os.getcwd(),
        "statistics",
        "",
    )
    os.makedirs(
        dir_statistics,
        exist_ok=True,
    )

    dask_client = Client(
        LocalCluster(
            n_workers=1,
            threads_per_worker=4,
            memory_limit=0,
            processes=True,
        )
    )

    list_dataset_statistics = [
        (DatasetContentWiseImpressions, StatisticsContentWiseImpressions),
        (DatasetMINDLarge, StatisticsMINDLarge),
        (DatasetFINNNoSlates, StatisticsFINNNoSlates),
        (DatasetRL4RS, StatisticsRL4RS),
        (DatasetYahooR6, StatisticsYahooR6A),
        (DatasetYahooR6, StatisticsYahooR6B),
        (DatasetSearchAds, StatisticsSearchAds),
        (DatasetPANDOR, StatisticsPANDOR),
        (DatasetAliCCP, StatisticsAliCCP),
        (DatasetAlimama, StatisticsAlimama),
        (DatasetComboFashion, StatisticsComboFashionCrossShops),
        (DatasetComboFashion, StatisticsComboFashionInShops),
        (DatasetKwaiFair, StatisticsKwaiFairSystem),
        (DatasetKwaiFair, StatisticsKwaiFairExperiment),
    ]
    list_statistics: list[dict] = []
    try:
        for dataset_class, dataset_statistics_class in list_dataset_statistics:
            st = time.time()
            dataset = dataset_class(
                dir_datasets=dir_datasets,
            )
            # item_in_slate, user, clicks, slates = dataset.issue
            statistics = dataset_statistics_class(
                dataset=dataset,
                dir_statistics=dir_statistics,
            )
            stats = statistics.statistics()
            print(stats)

            endst = time.time()
            stats["time"] = endst - st

            list_statistics.append(stats)

        print(list_statistics)
    except Exception as exception:
        with open(
            os.path.join(dir_statistics, "fail_datasets.txt"),
            "w",
        ) as file_exception:
            print("Failed")
            traceback.print_exception(
                exception,
                file=file_exception,
            )
    finally:
        df_results = pd.DataFrame.from_records(list_statistics)
        print("finally: \n", df_results)

        df_results.to_csv(
            path_or_buf=os.path.join(dir_statistics, "datasets.csv"),
            sep=";",
            encoding="utf-8",
            index=False,
        )

    dask_client.close()
