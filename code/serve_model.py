"""
Provides a REST API for the model.
"""

import os
from typing import Any, Dict, List

import joblib  # type: ignore
import train_classifier as model
from fastapi import FastAPI
from load_data import load_training_data
from pydantic import BaseModel

app = FastAPI()


class Query(BaseModel):
    """A query."""

    title: str


# TODO get classifier, embedding & mlb here
clf, emb = joblib.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "output", "clf.joblib"
    )
)
_, _, mlb = load_training_data()


@app.post("/predict/")
def predict(query: Query) -> Dict[str, Any]:
    """Prediction endpoint."""
    # TODO log
    return {"labels": model.predict(query.title, clf, emb, mlb)[0]}
