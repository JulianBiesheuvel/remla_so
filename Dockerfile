FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# TODO upgrade pip?
RUN pip install -r requirements.txt

# COPY . ./

# TODO this is a mess
COPY src/load_data.py src/load_data.py
COPY src/serve_model.py src/serve_model.py
COPY src/train_classifier.py src/train_classifier.py
COPY data/train.joblib data/train.joblib
COPY output/clf.joblib output/clf.joblib

EXPOSE 8080

ENTRYPOINT [ "/bin/bash", "-c" ]
CMD [ "cd", "src", ";", "uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8080" ]