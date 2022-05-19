from functools import reduce
from typing import List

import numpy as np
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
)
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from typing_extensions import TypedDict

from src import conf, preprocess, utils

Score = TypedDict(
    "Score",
    {"accuracy": float, "f1": float, "ap": float, "recall": float, "roc": float},
)


class Embedding:
    def fit_transform(self, X):
        pass

    def transform(self, X):
        pass


class TFIDF(Embedding):
    def __init__(
        self, min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern="(\S+)", **kwargs
    ):
        self.emb = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            token_pattern=token_pattern,
            **kwargs
        )

    def fit_transform(self, X):
        return self.emb.fit_transform(X)

    def transform(self, X):
        return self.emb.transform(X)


class BagOfWords(Embedding):
    """Bag of words sentence embedding."""

    def __init__(self, size=5000):
        self.size = size
        self.word2idx = {}

    def fit(self, X: List[str]):
        """Fits the word index to the given sentences."""
        # count word occurrences
        words_counts = {}
        for x in X:
            for word in x.split():
                if word in words_counts:
                    words_counts[word] += 1
                else:
                    words_counts[word] = 1
        # compute index by occurrence count
        idx2word = sorted(words_counts, key=words_counts.get, reverse=True)[: self.size]
        # store reverse index
        self.word2idx = {word: i for i, word in enumerate(idx2word)}

    def fit_transform(self, X: List[str]):
        """Fits to the data and returns the embedded X vector."""
        # fit to the training data
        self.fit(X)
        # return the transformed data
        return self.transform(X)

    def _transform(self, x: str) -> np.ndarray:
        """Transforms a single sentence."""
        result_vector = np.zeros(self.size)

        for word in x.split():
            if word in self.word2idx:
                result_vector[self.word2idx[word]] += 1
        return result_vector

    def transform(self, X: List[str]):
        """Return the transformed vector."""
        return sp_sparse.vstack([sp_sparse.csr_matrix(self._transform(w)) for w in X])


class Model:
    """A model for predicting StackOverflow tags."""

    def __init__(
        self,
        embedding: Embedding = TFIDF(),
        penalty="l1",
        C=1,
    ):
        self.name = type(embedding).__name__
        self.embedding = embedding
        self.mlb = None
        self.clf = None
        self.penalty = penalty
        self.C = C

    def train(self, X, y):
        """Trains the model on the given training data and labels."""
        # binarize labels
        tags = reduce(lambda acc, ts: acc.union(ts), y, set())
        self.mlb = MultiLabelBinarizer(classes=list(tags))
        yb = self.mlb.fit_transform(y)
        # embed training data
        X_emb = self.embedding.fit_transform(X)
        # fit classifier
        reg = LogisticRegression(
            penalty=self.penalty, C=self.C, dual=False, solver="liblinear"
        )
        self.clf = OneVsRestClassifier(reg)
        self.clf.fit(X_emb, yb)

    def _predict(self, X: List[str]):
        """Preprocess input and predict tags using a pretrained model."""
        assert self.mlb != None and self.clf != None, "You need to train first"
        # preprocess
        preprocessed = preprocess.preprocess(X)
        # embed request
        X_emb = self.embedding.transform(preprocessed)
        # call trained model
        y_pred = self.clf.predict(X_emb)
        return y_pred

    def predict(self, X: List[str]):
        """Preprocess input and predict tags using a pretrained model."""
        y_pred = self._predict(X)
        return self.mlb.inverse_transform(y_pred)

    def eval(self, X: List[str], y: List[List[str]]) -> Score:
        """Evaluates the model performance given a validation set."""
        y_pred = self._predict(X)
        y_true = self.mlb.transform(y)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted"),
            "ap": average_precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, labels=None, average="macro"),
            "roc": roc_auc(y_true, y_pred, multi_class="ovo"),
        }

    @staticmethod
    def load(model: str):
        return utils.load(conf.MODEL_DIR, model + ".joblib")
