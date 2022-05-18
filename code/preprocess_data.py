from functools import reduce

import nltk

nltk.download("stopwords")
import re
from ast import literal_eval

import numpy as np
import pandas as pd
from joblib import dump, load
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer


def read_raw_data(filename):
    data = pd.read_csv(filename, sep="\t")
    data["tags"] = data["tags"].apply(literal_eval)
    return data


REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOPWORDS = set(stopwords.words("english"))


def text_prepare(text):
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


def main():
    train = read_raw_data("data/train.tsv")
    validation = read_raw_data("data/validation.tsv")

    test = pd.read_csv("data/test.tsv", sep="\t")

    X_train, y_train = train["title"].to_numpy(), train["tags"].to_numpy()
    X_val, y_val = validation["title"].to_numpy(), validation["tags"].to_numpy()
    X_test = test["title"].to_numpy()

    X_train = [text_prepare(x) for x in X_train]
    X_val = [text_prepare(x) for x in X_val]
    X_test = [text_prepare(x) for x in X_test]

    all_tags = reduce(lambda acc, ts: acc.union(ts), y_train, set())

    # tags_counts = {}
    # for tags in y_train:
    #     for tag in tags:
    #         if tag in tags_counts:
    #             tags_counts[tag] += 1
    #         else:
    #             tags_counts[tag] = 1

    # mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    mlb = MultiLabelBinarizer(classes=list(all_tags))
    y_train = mlb.fit_transform(y_train)
    dump((X_train, y_train, mlb), "data/train.joblib")
    y_val = mlb.fit_transform(y_val)
    dump((X_val, y_val, mlb), "data/val.joblib")
    dump(X_test, "data/test.joblib")


if __name__ == "__main__":
    main()
