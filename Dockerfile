# 1. export poetry requirements and run dvc repro 
#    to make sure that the model artifacts exist
FROM python:3.9-slim as build

WORKDIR /build

RUN apt update \
  #  && apt install -y gcc g++ \
  && pip install poetry

COPY poetry.lock pyproject.toml /build/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
#  && pip install --no-cache-dir --upgrade 'dvc[gdrive]'
#  && poetry export -f requirements.txt --dev --output all-requirements.txt --without-hashes \
#  && pip install --no-cache-dir --upgrade -r all-requirements.txt

# COPY ./ /build/

# we need the DVC versioned model files
# RUN dvc pull train

# 3. install requirements, copy code & models and package api
FROM python:3.9-slim

WORKDIR /app

COPY --from=build /build/requirements.txt requirements.txt

RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY src src
COPY output/models output/models

# COPY --from=build /build/output/models output/models

EXPOSE 8080

# multiple gunicorn workers
# CMD ["gunicorn", "src.service:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]

# single uvicorn worker (for k8s as process manager)
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]