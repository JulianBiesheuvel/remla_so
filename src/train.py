"""
Code for training the models.

Needs to be in a separate file to not mess up the joblib file.
"""

from src import conf, model, utils


def main() -> None:
    # load data
    train = utils.load(conf.PROCESSED_DATA_DIR, "train.joblib")
    val = utils.load(conf.PROCESSED_DATA_DIR, "validation.joblib")
    #  - fit embedding & train classifier
    m = model.Model()
    m.train(*train)
    utils.store(m, conf.MODEL_DIR, m.name + ".joblib")
    # TODO store?
    print(m.eval(*val))

    m2 = model.Model(embedding=model.BagOfWords(5000))
    m2.train(*train)
    utils.store(m2, conf.MODEL_DIR, m2.name + ".joblib")
    print(m2.eval(*val))


if __name__ == "__main__":
    main()
