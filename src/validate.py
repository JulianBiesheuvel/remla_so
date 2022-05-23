""" Module for validating new data and creating new training/test splits. """

import os
from ast import literal_eval
from typing import AnyStr, List, Tuple

import pandas as pd
import tensorflow_data_validation as tfdv
from sklearn.model_selection import train_test_split

from src import conf, utils


def gen_stats(
    dataframe: pd.DataFrame, name: AnyStr
) -> Tuple[List[str], List[List[str]]]:
    """
    Generate tfdv stats for a dataframe and return the contents of the dataframe.
    """
    stats = tfdv.generate_statistics_from_dataframe(dataframe)
    tfdv.write_stats_text(stats, os.path.join(conf.DATA_STATS_DIR, name + ".txt"))  # type: ignore
    X, y = (
        dataframe["title"].to_numpy(),
        dataframe["tags"].apply(literal_eval).to_numpy(),
    )
    return X, y


def main() -> None:
    """
    Splits the raw processed data into training/validation and generates tfdv statistics.
    """
    dataframe = pd.read_csv(os.path.join(conf.PROCESSED_DATA_DIR, "preprocessed.csv"))
    train, test = train_test_split(dataframe, random_state=42)
    train = gen_stats(train, "train")
    validation = gen_stats(test, "validation")

    utils.store(
        train,
        conf.PROCESSED_DATA_DIR,
        "train.joblib",
    )

    utils.store(
        validation,
        conf.PROCESSED_DATA_DIR,
        "validation.joblib",
    )

    # How to visualize this? Maybe in the api or Grafana?
    # tfdv.visualize_statistics(rhs_statistics=train_stats, lhs_statistics=test_stats, rhs_name="TRAIN", lhs_name="TEST")


if __name__ == "__main__":
    main()
