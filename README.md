# Multilabel classification on Stack Overflow tags
Predict tags for posts from StackOverflow with multilabel classification approach.

## Dataset
- Dataset of post titles from StackOverflow

## Transforming text to a vector
- Transformed text data to numeric vectors using bag-of-words and TF-IDF.

## MultiLabel classifier
[MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) to transform labels in a binary form and the prediction will be a mask of 0s and 1s.

[Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for Multilabel classification
- Coefficient = 10
- L2-regularization technique

## Evaluation
Results evaluated using several classification metrics:
- [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)

## Libraries
- [Numpy](http://www.numpy.org/) — a package for scientific computing.
- [Pandas](https://pandas.pydata.org/) — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
- [scikit-learn](http://scikit-learn.org/stable/index.html) — a tool for data mining and data analysis.
- [NLTK](http://www.nltk.org/) — a platform to work with natural language.

Note: this sample project was originally created by @partoftheorigin

## UPDATE `poetry`

Since `mllint` wants us to use `poetry`, stuff became messed up XD

Use `poetry install` to install the project dependencies

Use `poetry run mllint` to run mllint

Use `poetry run dvc repro` to run `dvc`

Use `poetry run pytest --junitxml=reports/tests.xml --cov=src --cov-report=xml:reports/coverage.xml` to run `pytest`

Use `poetry run python -m src.*` to `preprocess`,`train`, `service`

To develop, use `poetry shell` to activate the environment.
Then you can use `mllint`, `pytest`, `python`, `dvc` as you would expect...

## Running the server

```
(cd src ; uvicorn serve_model:app --reload)
```

## Notes

There is a python core module named `code`. Shadowing mostly works as one would expect, but `gunicorn` says no.
Using `joblib`/`pickle` to save an object defined in the same file will mess with the stored path.
Again `gunicorn` and `uvicorn` don't play well with this, hence the training is done in a separate file.

So even the tests don't work with this...
`ImportError: cannot import name '__version__' from 'code' (/usr/lib/python3.8/code.py)`
FYI this is the module messing with the naming [https://docs.python.org/3.9/library/code.html](https://docs.python.org/3.9/library/code.html)

Lessons learned
- python naming & shadowing is shady and established packages (`uvicorn`,`gunicorn`)
- don't use code as your source code directory name if you have to reference it anywhere