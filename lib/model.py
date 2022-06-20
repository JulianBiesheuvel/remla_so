"""
The embeddings and model for the ml task.
"""
# pylint: skip-file

from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Dict, List, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
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

from lib import conf, preprocess, utils

Score = TypedDict(
    "Score",
    {"accuracy": float, "f1": float, "ap": float, "recall": float, "roc": float},
)


class Embedding(ABC):
    """Common interface for the embedding functionality."""

    @abstractmethod
    def fit_transform(self, X: List[str]) -> npt.NDArray[np.float64]:
        """Should fit embedding and return the embedded data."""
        return NotImplemented

    @abstractmethod
    def transform(self, X: List[str]) -> npt.NDArray[np.float64]:
        """Returns the embedded data."""
        return NotImplemented


class TFIDF(Embedding):
    """Wrapper for the scikit TfidfVectorizer class."""

    def __init__(  # nosec
        self,
        min_df: float = 5,
        max_df: float = 0.9,
        ngram_range: Tuple[int, int] = (1, 2),
        token_pattern: str = r"(\S+)",
        **kwargs: Dict[str, Any]
    ) -> None:
        self.emb = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            token_pattern=token_pattern,
            **kwargs
        )

    def fit_transform(self, X: List[str]) -> npt.NDArray[np.float64]:
        """Fits to the data and returns the embedded X vector."""
        return cast(npt.NDArray[np.float64], self.emb.fit_transform(X))

    def transform(self, X: List[str]) -> npt.NDArray[np.float64]:
        """Return the transformed vector."""
        return cast(npt.NDArray[np.float64], self.emb.transform(X))


class BagOfWords(Embedding):
    """Bag of words sentence embedding."""

    def __init__(self, size: int = 5000) -> None:
        self.size: int = size
        self.word2idx: Dict[str, int] = {}

    def fit(self, X: List[str]) -> None:
        """Fits the word index to the given sentences."""
        # count word occurrences
        words_counts: Dict[str, int] = {}
        for x in X:
            for word in x.split():
                if word in words_counts:
                    words_counts[word] += 1
                else:
                    words_counts[word] = 1
        # compute index by occurrence count
        idx2word = sorted(words_counts, key=words_counts.get, reverse=True)[: self.size]  # type: ignore
        # store reverse index
        self.word2idx = {word: i for i, word in enumerate(idx2word)}

    def fit_transform(self, X: List[str]) -> npt.NDArray[np.float64]:
        """Fits to the data and returns the embedded X vector."""
        # fit to the training data
        self.fit(X)
        # return the transformed data
        return self.transform(X)

    def _transform(self, x: str) -> npt.NDArray[np.float64]:
        """Transforms a single sentence."""
        result_vector = np.zeros(self.size)

        for word in x.split():
            if word in self.word2idx:
                result_vector[self.word2idx[word]] += 1
        return result_vector

    def transform(self, X: List[str]) -> npt.NDArray[np.float64]:
        """Return the transformed vector."""
        return cast(
            npt.NDArray[np.float64],
            sp_sparse.vstack([sp_sparse.csr_matrix(self._transform(w)) for w in X]),
        )


class Model:
    """A model for predicting StackOverflow tags."""

    def __init__(
        self,
        embedding: Embedding = TFIDF(),
        penalty: str = "l1",
        C: float = 1,
    ) -> None:
        self.name = type(embedding).__name__
        self.embedding = embedding
        self.mlb: Union[None, MultiLabelBinarizer] = None
        self.clf: Union[None, OneVsRestClassifier] = None
        self.penalty = penalty
        self.C = C

    def train(self, X: List[str], y: List[List[str]]) -> None:
        """Trains the model on the given training data and labels."""
        # binarize labels
        tags = reduce(lambda acc, ts: acc.union(ts), y, set())  # type: ignore
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

    def _predict(self, X: List[str]) -> List[List[float]]:
        """Preprocess input and predict tags using a pretrained model."""
        assert (  # nosec
            self.mlb is not None and self.clf is not None
        ), "You need to train first"  # nosec
        # preprocess
        preprocessed = preprocess.preprocess(X)
        # embed request
        X_emb = self.embedding.transform(preprocessed)
        # call trained model
        y_pred = cast(List[List[float]], self.clf.predict(X_emb))
        return y_pred

    def predict(self, X: List[str]) -> List[List[str]]:
        """Preprocess input and predict tags using a pretrained model."""
        y_pred = self._predict(X)
        return cast(List[List[str]], self.mlb.inverse_transform(y_pred))  # type: ignore

    def eval(
        self,
        X: List[str],
        y: List[List[str]],
        include_scores: List[str] = ["all"],
    ) -> Score:
        """Evaluates the model performance given a validation set."""
        y_pred = self._predict(X)
        y_true = self.mlb.transform(y)  # type: ignore
        return {
            "accuracy": accuracy_score(y_true, y_pred)
            if "accuracy" in include_scores or "all" in include_scores
            else -1,
            "f1": f1_score(y_true, y_pred, average="weighted")
            if "f1" in include_scores or "all" in include_scores
            else -1,
            "ap": average_precision_score(y_true, y_pred, average="macro")
            if "ap" in include_scores or "all" in include_scores
            else -1,
            "recall": recall_score(y_true, y_pred, labels=None, average="macro")
            if "recall" in include_scores or "all" in include_scores
            else -1,
            "roc": roc_auc(y_true, y_pred, multi_class="ovo")
            if "roc" in include_scores or "all" in include_scores
            else -1,
        }

    @staticmethod
    def load(model: str) -> "Model":
        """Helper for loading stored models."""
        return cast(Model, utils.load(conf.MODEL_DIR, model + ".joblib"))
