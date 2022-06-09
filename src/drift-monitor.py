"""
Provides a REST API for the model.
"""

# pylint: disable=E0611,R0903

from time import time
from typing import List

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.responses import RedirectResponse
from skmultiflow.drift_detection.adwin import ADWIN
import psycopg

from src.model import Model

def get_new_data(batch_size: int = 20000):
    """Retrieves new testing data"""
    X,y = [], []
    with psycopg.connect() as conn:
        with conn.cursor() as cur:
            for title, tags in cur.execute("select title, tags from questions order by created_at desc limit %s;", (batch_size, )):
                X.append(title)
                y.append(eval(tags))
        conn.close()
    return (X,y)

m = Model.load("TFIDF")
def calculate_scores():
    X, y = get_new_data()
    return m.eval(X, y)

class DriftMonitor:
    """Class for monitoring drift over time."""
    def __init__(self,
        metrics=List[str],
        detector = ADWIN,
        calculate_scores = calculate_scores
    ) -> None:
        self.metrics = metrics
        self.detectors = { m: detector() for m in metrics }
        self.last_scores = {}
        self.calculate_score = calculate_scores
    
    def tick(self):
        """Detects prediction drift"""
        scores = self.calculate_scores()
        for metric in self.metrics:
            self.detectors[metric].add_element(scores[metric])
        self.last_scores = scores

    def prometheus_exporter(self):
        """Prometheus integration"""
        ts = int(time() * 1000)  # unix timestamp in [ms]

        return "\n".join("""\
# HELP Last batch {k} score
# TYPE drift_monitor_{k} gauge
drift_monitor_{k} {v} {ts}  

# HELP {k} drift detected
# TYPE drift_monitor_drift_detected_{k} gauge
drift_monitor_drift_detected_{k} {drift} {ts}

# HELP 1 if {k} is in the drift warning zone
# TYPE drift_monitor_drift_warning_{k} gauge
drift_monitor_drift_warning_{k} {warn} {ts}

""".format(
    ts=ts,
    k=k,
    v=v,
    drift=1 if self.detectors[k].detected_change else 0,
    warn=1 if self.detectors[k].detected_warning_zone else 0,
) for k,v in self.last_scores)

drift_monitor = DriftMonitor(
    metrics = ["accuracy", "f1"],
    detector=ADWIN,
    calculate_scores=calculate_scores
)

app = FastAPI()

@app.get("/")
def index() -> RedirectResponse:
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")

@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics() -> str:
    """Returns Prometheus text-based metrics."""
    drift_monitor.tick()
    return drift_monitor.prometheus_exporter()
