import numpy as np
from joblib import dump, load
from load_data import load_test_data, load_training_data, load_validation_data
from sklearn.feature_extraction.text import TfidfVectorizer


def my_bag_of_words(text, words_to_index, dict_size):
    """
    text: a string
    dict_size: size of the dictionary

    return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)

    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


# def train_embedding(data):
#     tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1,2), token_pattern='(\S+)')
#     X_train = tfidf_vectorizer.fit_transform(X_train)


def tfidf_features(X_train, X_val, X_test):
    """
    X_train, X_val, X_test — samples
    return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(
        min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern="(\S+)"
    )

    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


def main():
    X_train, y_train = load_training_data()
