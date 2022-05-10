from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score as roc_auc
from train_classifier import train_classifier
from joblib import load
from sys import argv
import os

from load_data import load_test_data, load_validation_data

def evaluation_scores(y_val, predicted):
    return {
    'accuracy' : accuracy_score(y_val, predicted),
    'f1' : f1_score(y_val, predicted, average='weighted'),
    'ap' : average_precision_score(y_val, predicted, average='macro'),
    'roc' : roc_auc(y_val, predicted, multi_class='ovo')
    }

def evaluate(type_clf='default'):
    x_val, y_val, _ = load_validation_data()

    stored_model, _ = load(type_clf + '.joblib')
    predictions = stored_model.predict(x_val)
    return evaluation_scores(y_val, predictions)

    # test_pred_inversed = mlb.inverse_transform(test_predictions)
    # test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))


def main():
    model_path = ''
    classifier = argv[1]
    scores = evaluate(os.path.join(model_path, classifier))
    model_scores = {
        'model' : argv[1],
        'scores' : scores
    }

    # TODO: Log results somewhere

if __name__ == '__main__':
    main()