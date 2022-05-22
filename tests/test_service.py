# pylint: disable=all
from fastapi.testclient import TestClient

from src.service import app

client = TestClient(app)


def test_predict() -> None:
    response = client.post(
        "/predict/", json={"title": "How do I invert a binary tree in python?"}
    )
    assert response.status_code == 200  # nosec
    # assert response.json() == {...}
