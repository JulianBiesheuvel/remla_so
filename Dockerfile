FROM python:3.9-slim as requirements

WORKDIR /tmp

RUN pip install poetry

COPY poetry.lock pyproject.toml /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.9-slim

WORKDIR /app

COPY --from=requirements /tmp/requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY src src
COPY output/models output/models

EXPOSE 8080

# multiple gunicorn workers
# CMD ["gunicorn", "src.service:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]

# single uvicorn worker 
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]