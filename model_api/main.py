"""
Provides a REST API for the model.
"""

# pylint: disable=E0611,R0903

from functools import reduce
from time import time
from typing import Dict, List, TypedDict

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Extra, conlist
from starlette.responses import RedirectResponse

from src.model import Model

app = FastAPI()


class Query(BaseModel, extra=Extra.forbid):
    """A single query."""

    query: str


class Queries(BaseModel, extra=Extra.forbid):
    """Multiple queries.

    Must contain at least one query and at most 42 queries.
    """

    queries: conlist(str, min_items=1, max_items=42)  # type: ignore


m = Model.load("TFIDF")

Metrics = TypedDict(
    "Metrics",
    {
        "num_requests": int,
        "num_batch_requests": int,
        "num_queries": int,
        "num_tags_total": int,
        "num_invalid_requests": int,
    },
)

metrics: Metrics = {
    "num_requests": 0,
    "num_batch_requests": 0,
    "num_queries": 0,
    "num_tags_total": 0,
    "num_invalid_requests": 0,
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Hook into FastAPI to get them metrics."""
    metrics["num_invalid_requests"] += 1
    return await request_validation_exception_handler(request, exc)


@app.get("/")
def index() -> RedirectResponse:
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


@app.post("/predict/")
def post_predict(query: Query) -> Dict[str, List[str]]:
    """Single prediction endpoint."""

    metrics["num_requests"] += 1
    metrics["num_queries"] += 1

    labels = m.predict([query.query])[0]

    metrics["num_tags_total"] += len(labels)

    return {"labels": labels}


@app.post("/predictions/")
def post_predictions(queries: Queries) -> Dict[str, List[List[str]]]:
    """Endpoint for multiple predictions."""

    metrics["num_requests"] += 1
    metrics["num_queries"] += len(queries.queries)
    metrics["num_batch_requests"] += 1

    labels = m.predict(queries.queries)

    metrics["num_tags_total"] += reduce(lambda acc, ls: acc + len(ls), labels, 0)

    return {"labels": labels}


@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics() -> str:
    """Returns Prometheus text-based metrics."""
    ts = int(time() * 1000)  # unix timestamp in [ms]

    return """\
# HELP num_requests The number of requests received so far
# TYPE num_requests counter
num_requests {num_requests} {ts}

# HELP num_batch_requests The number of batch requests received so far
# TYPE num_batch_requests counter
num_batch_requests {num_batch_requests} {ts}

# HELP num_queries The number of queries received so far
# TYPE num_queries counter
num_queries {num_queries} {ts}

# HELP num_tags_total The number of tags returned so far
# TYPE num_tags_total counter
num_tags_total {num_tags_total} {ts}

# HELP num_invalid_requests The total number of invalid requests so far
# TYPE num_invalid_requests counter
num_invalid_requests {num_invalid_requests} {ts}
""".format(
        **metrics, ts=ts
    )
