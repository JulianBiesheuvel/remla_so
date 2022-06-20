"""
Periodically queries the SO api for new data and provides prometheus metrics.
"""
# pylint: skip-file
# mypy: ignore-errors

from datetime import datetime
from time import time

import psycopg
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.responses import RedirectResponse

metrics = {
    "num_questions_retrieved": 0,
    "status": 200,
    "num_requests": 0,
    "num_successful": 0,
    "quota_remaining": 0,
}


def query_so() -> None:
    """Retrieves new questions using the stackoverflow api"""
    metrics["num_requests"] += 1
    # get stackoverflow data
    r = requests.get(
        "https://api.stackexchange.com/2.3/questions?pagesize=100&order=desc&sort=creation&site=stackoverflow"
    )

    metrics["status"] = r.status_code
    if r.status_code == 200:
        data = r.json()
        metrics["quota_remaining"] = data["quota_remaining"]
        metrics["num_questions_retrieved"] += len(data["items"])
        # store data
        prepared_data = [
            (
                q["question_id"],
                q["title"],
                str(q["tags"]),
                datetime.fromtimestamp(q["creation_date"]),
            )
            for q in data["items"]
        ]
        with psycopg.connect() as conn:  # pylint: ignore
            with conn.cursor() as cur:
                cur.executemany(
                    "insert into questions (id, title, tags, created_at) values (%s, %s, %s, %s) on conflict do nothing",
                    prepared_data,
                )
            conn.commit()
            conn.close()
        metrics["num_successful"] += 1


app = FastAPI()


@app.on_event("startup")
def startup() -> None:
    """Starts the background job"""
    scheduler = BackgroundScheduler({"apscheduler.timezone": "UTC"})
    scheduler.add_job(query_so, "interval", seconds=288)  # 1day / 288s = 300requests
    scheduler.start()


@app.get("/")
def index() -> RedirectResponse:
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics() -> str:
    """Returns Prometheus text-based metrics."""
    ts = int(time() * 1000)  # unix timestamp in [ms]

    return """\
# HELP so_agent_num_requests The number of requests made so far
# TYPE so_agent_num_requests counter
so_agent_num_requests {num_requests} {ts}

# HELP so_agent_num_questions_retrieved The number of SO questions retrieved so far
# TYPE so_agent_num_questions_retrieved counter
so_agent_num_questions_retrieved {num_questions_retrieved} {ts}

# HELP so_agent_num_successful The number of successful requests made so far
# TYPE so_agent_num_successful counter
so_agent_num_successful {num_successful} {ts}

# HELP so_agent_last_http_status The HTTP status of the last request
# TYPE so_agent_last_http_status gauge
so_agent_last_http_status {status} {ts}

# HELP so_agent_quota_remaining The number of requests remaining in the SO api quota
# TYPE so_agent_quota_remaining gauge
so_agent_quota_remaining {quota_remaining} {ts}
""".format(
        **metrics, ts=ts
    )
