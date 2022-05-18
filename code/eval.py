"""
    Runs evaluation on the provided model
"""

import os
from sys import argv

from joblib import load  # type: ignore
from load_data import load_training_data, load_validation_data
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
)
from sklearn.metrics import roc_auc_score as roc_auc
from train_classifier import predict


def evaluation_scores(y_val, predicted):
    return {
        "accuracy": accuracy_score(y_val, predicted),
        "f1": f1_score(y_val, predicted, average="weighted"),
        "ap": average_precision_score(y_val, predicted, average="macro"),
        "recall": recall_score(y_val, predicted, labels=None, average="macro"),
        "roc": roc_auc(y_val, predicted, multi_class="ovo"),
    }


def evaluate(classifier, embedding):
    x_val, y_val, _ = load_validation_data()
    X_emb = embedding.transform(x_val)
    labels = classifier.predict(X_emb)
    return evaluation_scores(y_val, labels)


def main():
    classifier = argv[1]
    clf, emb = load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "output",
            f"{classifier}.joblib",
        )
    )
    scores = evaluate(clf, emb)
    model_scores = {"model": argv[1], "scores": scores}
    print(model_scores)
    # TODO: Log results somewhere


if __name__ == "__main__":
    main()
