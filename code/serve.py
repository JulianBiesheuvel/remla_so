"""
Provides a REST API for the model.
"""
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from . import model

m = model.Model.load("TfidfVectorizer")

app = FastAPI()


class Query(BaseModel):
    """A query."""

    title: str


@app.post("/predict/")
def predict(query: Query) -> Dict[str, Any]:
    """Prediction endpoint."""
    # TODO log
    return {"labels": m.predict([query.title])[0]}


if __name__ == '__main__':
    print(m.predict(["iphone"]))