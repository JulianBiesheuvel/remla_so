FROM python:3.9-slim as build

WORKDIR /build

RUN apt update \
  && pip install poetry

COPY poetry.lock pyproject.toml /build/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# As Scrapy runs on Python, I choose the official Python 3 Docker image.
FROM python:3.9-slim

# Set the working directory to /usr/src/app.
WORKDIR /app

# Copy the file from the local host to the filesystem of the container at the working directory.
COPY --from=build build/requirements.txt requirements.txt

# Install Scrapy specified in requirements.txt.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the project source code from the local host to the filesystem of the container at the working directory.
COPY . .

# Run the crawler when the container launches.
CMD [ "scrapy", "crawl", "questions" ]

