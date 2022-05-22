"""
Provides a REST API for the model.
"""

# pylint: disable=E0611,R0903

from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from src.model import Model

app = FastAPI()


@app.get("/")
def index():
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


class Query(BaseModel):
    """A query."""

    title: str


m = Model.load("TFIDF")


@app.post("/predict/")
def predict(query: Query) -> Dict[str, Any]:
    """Prediction endpoint."""

    labels = m.predict([query.title])[0]
    # TODO log query + labels ?
    return {"labels": labels}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serve:app", port=8000)  # nosec
