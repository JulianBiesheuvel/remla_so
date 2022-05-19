"""
Provides a REST API for the model.
"""
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from src.model import Model

app = FastAPI()


class Query(BaseModel):
    """A query."""

    title: str


m = Model.load("TFIDF")


@app.post("/predict/")
def predict(query: Query) -> Dict[str, Any]:
    """Prediction endpoint."""
    # TODO log
    return {"labels": m.predict([query.title])[0]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serve:app", port=8000)  # nosec
