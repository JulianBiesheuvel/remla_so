import re
from functools import reduce

import nltk

nltk.download("stopwords")

import os
from ast import literal_eval
from typing import List, Tuple

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm

from src import conf, utils

REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOPWORDS = set(stopwords.words("english"))


def preprocess_one(text: str) -> str:
    """
    text: a string

    return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = re.sub(
        REPLACE_BY_SPACE_RE, " ", text
    )  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(
        BAD_SYMBOLS_RE, "", text
    )  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join(
        [word for word in text.split() if not word in STOPWORDS]
    )  # delete stopwords from text
    return text


preprocess = np.vectorize(preprocess_one)


def preprocess_raw(filename: str) -> Tuple[List[str], List[List[str]]]:
    """Preprocesses a .tsv file."""
    # read data
    data = pd.read_csv(os.path.join(conf.RAW_DATA_DIR, filename), sep="\t")
    X, y = data["title"].to_numpy(), data["tags"].apply(literal_eval).to_numpy()
    # return preprocessed data
    return preprocess(X), y


def main() -> None:
    """Preprocesses the test and validation set."""
    for dataset in tqdm(os.listdir(conf.RAW_DATA_DIR)):
        preprocessed = preprocess_raw(dataset)
        utils.store(
            preprocessed,
            conf.PROCESSED_DATA_DIR,
            os.path.splitext(dataset)[0] + ".joblib",
        )

    # test = pd.read_csv("test.tsv", sep="\t")
    # test = prepare(test["title"].to_numpy())
    # dump(test, os.path.join(conf.DATA_DIR, "test.joblib"))


if __name__ == "__main__":
    main()
