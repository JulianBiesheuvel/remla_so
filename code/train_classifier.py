import os

import numpy as np
from joblib import dump
from load_data import load_test_data, load_training_data, load_validation_data
from preprocess_data import text_prepare
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class BagOfWords:
    def _init_(self, words_counts, X, dict_size=5000):
        self.DICT_SIZE = dict_size
        self.INDEX_TO_WORDS = sorted(words_counts, key=words_counts.get, reverse=True)[
            : self.DICT_SIZE
        ]
        self.WORDS_TO_INDEX = {word: i for i, word in enumerate(self.INDEX_TO_WORDS)}
        self.ALL_WORDS = self.WORDS_TO_INDEX.keys()
        self.most_common_tags = None
        self.most_common_words = None

    def fit_transform(self, X_train, X_val, X_test):
        """Fits to the data and returns the embedded X vector"""
        X_train_mybag = sp_sparse.vstack(
            [
                sp_sparse.csr_matrix(
                    self.transform(text, self.WORDS_TO_INDEX, self.DICT_SIZE)
                )
                for text in X_train
            ]
        )
        X_val_mybag = sp_sparse.vstack(
            [
                sp_sparse.csr_matrix(
                    self.transform(text, self.WORDS_TO_INDEX, self.DICT_SIZE)
                )
                for text in X_val
            ]
        )
        X_test_mybag = sp_sparse.vstack(
            [
                sp_sparse.csr_matrix(
                    self.transform(text, self.WORDS_TO_INDEX, self.DICT_SIZE)
                )
                for text in X_test
            ]
        )

        return X_train_mybag, X_val_mybag, X_test_mybag

    def transform(self, X, dict_size, text, words_to_index):
        result_vector = np.zeros(dict_size)

        for word in text.split():
            if word in words_to_index:
                result_vector[words_to_index[word]] += 1
        return result_vector

    def get_most_common_words(self, X_train, y_train):

        # Dictionary of all tags from train corpus with their counts.
        tags_counts = {}
        # Dictionary of all words from train corpus with their counts.
        words_counts = {}

        for sentence in X_train:
            for word in sentence.split():
                if word in words_counts:
                    words_counts[word] += 1
                else:
                    words_counts[word] = 1

        for tags in y_train:
            for tag in tags:
                if tag in tags_counts:
                    tags_counts[tag] += 1
                else:
                    tags_counts[tag] = 1

        self.most_common_tags = sorted(
            tags_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        self.most_common_words = sorted(
            words_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]


def train_classifier(
    X_train,
    y_train,
    penalty="l1",
    C=1,
    embedding=TfidfVectorizer(
        min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern="(\S+)"
    ),
):
    """
    X_train, y_train — training data

    return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    X_emb = embedding.fit_transform(X_train)

    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver="liblinear")
    clf = OneVsRestClassifier(clf)
    clf.fit(X_emb, y_train)

    return clf, embedding


def main():
    # training

    # load data
    X_train, y_train, mlb = load_training_data()
    #  - fit embedding & train classifier
    clf, emb = train_classifier(X_train, y_train)
    #  - dump embedding + classifier
    os.makedirs("output/clf.joblib", exist_ok=True)
    dump((clf, emb), "output/clf.joblib")

    clf_bow, emb_bow = train_classifier(X_train, y_train, embedding=BagOfWords())
    os.makedirs("output/clf_bow.joblib", exist_ok=True)
    dump((clf_bow, emb_bow), "output/clf_bow.joblib")

    # TODO BAGofWords
    # TODO see how it performs on validation data


def predict(x, clf, emb, mlb):
    """Predict tags using a pretrained model."""
    # preprocess
    preprocessed = text_prepare(x)
    # embed request
    X_emb = emb.transform([preprocessed])
    # call trained model
    labels = clf.predict(X_emb)
    scores, *_ = clf.decision_function(X_emb)
    # restore labels
    lab, *_ = mlb.inverse_transform(labels)
    return lab, scores


if __name__ == "__main__":
    main()

    # print(predict('How to parse XML file in php?'))
