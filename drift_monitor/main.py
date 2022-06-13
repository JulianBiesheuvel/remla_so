"""
Provides a REST API for the model.
"""

from ast import literal_eval

import psycopg
from apscheduler.schedulers.background import BackgroundScheduler

# pylint: disable=E0611,R0903
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from skmultiflow.drift_detection.adwin import ADWIN
from starlette.responses import RedirectResponse

from drift_monitor.drift_monitor import DriftMonitor
from lib.model import Model


def get_new_data(batch_size: int = 20000):
    """Retrieves new testing data"""
    X, y = [], []
    with psycopg.connect() as conn:  # pylint: ignore
        with conn.cursor() as cur:
            for title, tags in cur.execute(
                "select title, tags from questions order by created_at desc limit %s;",  # nosec
                (batch_size,),
            ):
                X.append(title)
                y.append(literal_eval(tags))
        conn.close()
    return (X, y)


m = Model.load("TFIDF")

# some of the others don't behave nicely
INCLUDE_SCORES = ["accuracy"]

def calculate_scores():
    """Gets data and calculates the model scores"""
    X, y = get_new_data()
    return m.eval(X, y, include_scores=INCLUDE_SCORES)


def load_model_metrics():
    """Loads the initial model metrics"""
    import json  # pylint: ignore
    import os  # pylint: ignore
    with open(os.path.join("output", "TFIDF.json"), encoding="utf8") as f:
        metrics = json.load(f)
    return { k:v for k,v in metrics.items() if k in INCLUDE_SCORES }


drift_monitor = DriftMonitor(
    calculate_scores,
    metrics=load_model_metrics(),
    detector=ADWIN,
)

app = FastAPI()


@app.on_event("startup")
def startup():
    """Starts the monitor tick as background job"""
    scheduler = BackgroundScheduler({"apscheduler.timezone": "UTC"})
    scheduler.add_job(drift_monitor.tick, "interval", minutes=5) # TODO minutes=30)
    scheduler.start()


@app.get("/")
def index() -> RedirectResponse:
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics() -> str:
    """Returns Prometheus text-based metrics."""
    return drift_monitor.prometheus_metrics()
