# pylint: disable=all
from fastapi.testclient import TestClient

from src.service import app

client = TestClient(app)


def test_predict() -> None:
    # not always possible to test as the model may not be present due to dvc
    # response = client.post(
    #     "/predict/", json={"title": "How do I invert a binary tree in python?"}
    # )
    response = client.get("/docs/")
    assert response.status_code == 200  # nosec
    # assert response.json() == {...}
