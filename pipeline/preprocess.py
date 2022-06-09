"""
Preprocessing step of DVC pipeline
"""

import os
from ast import literal_eval

import pandas as pd

from lib import conf, utils
from lib.preprocess import preprocess


def preprocess_raw(filename: str) -> pd.DataFrame:
    """Preprocesses a .tsv file and returns the pandas DataFrame."""
    # read data
    data = pd.read_csv(os.path.join(conf.RAW_DATA_DIR, filename), sep="\t")
    X, y = data["title"].to_numpy(), data["tags"].apply(literal_eval).to_numpy()
    data["title"] = preprocess(X)
    data["tags"] = y
    # return preprocessed data
    return data


def main() -> None:
    """Preprocesses the test and validation set."""
    # for dataset in tqdm(os.listdir(conf.RAW_DATA_DIR)):
    dataframe = preprocess_raw(os.path.join(conf.RAW_DATA_DIR, "raw.tsv"))
    # utils.store(
    #     preprocessed,
    #     conf.PROCESSED_DATA_DIR,
    #     os.path.splitext(dataset)[0] + ".joblib",
    # )
    utils.store_dataframe(dataframe, conf.PROCESSED_DATA_DIR, "preprocessed.csv")

    # test = pd.read_csv("test.tsv", sep="\t")
    # test = prepare(test["title"].to_numpy())
    # dump(test, os.path.join(conf.DATA_DIR, "test.joblib"))


if __name__ == "__main__":
    main()
