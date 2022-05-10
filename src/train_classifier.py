from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
from sys import argv

import os
from load_data import load_test_data, load_training_data, load_validation_data
from preprocess_data import text_prepare


def train_classifier(X_train, y_train, penalty='l1', C=1, embedding=TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1,2), token_pattern='(\S+)')):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    X_emb = embedding.fit_transform(X_train)
    
    clf = LogisticRegression(penalty=penalty, C=C, dual=False, solver='liblinear')
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
    os.makedirs('output/clf.joblib', exist_ok=True)
    dump((clf, emb), 'output/clf.joblib')

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

if __name__ == '__main__':
    main()

    # print(predict('How to parse XML file in php?'))