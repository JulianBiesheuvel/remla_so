import os

import joblib
from load_data import load_training_data

import train_classifier as model

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

class PredictionReq(BaseModel):
    heading: str

# TODO get classifier, embedding & mlb here
clf, emb = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'clf.joblib'))
_, _, mlb = load_training_data()

@app.post("/predict/")
def predict(query: PredictionReq):
    # TODO log
    return {
        "labels": model.predict(query.heading, clf, emb, mlb)[0]
    }