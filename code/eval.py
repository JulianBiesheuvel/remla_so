import os
from sys import argv

from joblib import load
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
        "roc": roc_auc(y_val, predicted, multi_class="ovo"),
    }


def evaluate(classifier, embedding):
    x_val, y_val, mlb = load_validation_data()
    # predictions = [predict(x, classifier, embedding, mlb) for x in x_val]
    y_val = mlb.fit_transform(y_val)
    predictions = predict(x_val[0], classifier, embedding, mlb)
    # print(predictions)
    print(y_val[0])
    print(predictions[1])
    return evaluation_scores(y_val[0], predictions[1])


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
