"""
Data preprocessing for the model.
"""

import re
import nltk

nltk.download("stopwords")

import numpy as np
from nltk.corpus import stopwords


REPLACE_BY_SPACE_RE = re.compile(r"[/(){}\[\]\|@,;]")
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
