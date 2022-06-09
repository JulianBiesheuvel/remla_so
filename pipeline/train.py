"""
Code for training the models.

Needs to be in a separate file to not mess up the joblib file.
"""

import json
import os

from tqdm import tqdm

from lib import conf, model, utils


def main() -> None:
    """Trains the model with both embeddings and stores the result"""
    # load data
    train = utils.load(conf.PROCESSED_DATA_DIR, "train.joblib")
    val = utils.load(conf.PROCESSED_DATA_DIR, "validation.joblib")

    #  - fit embeddings & train classifiers
    for emb in tqdm([model.TFIDF(), model.BagOfWords(5000)]):
        m = model.Model(embedding=emb)
        m.train(*train)
        utils.store(m, conf.MODEL_DIR, m.name + ".joblib")

        eval_report = m.eval(*val)

        with open(
            os.path.join(conf.OUTPUT_DIR, m.name + ".json"), "w+", encoding="utf-8"
        ) as f:
            json.dump(eval_report, f)
        print(eval_report)


if __name__ == "__main__":
    main()
