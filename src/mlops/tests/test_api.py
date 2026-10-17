import pytest
from fastapi.testclient import TestClient

from src.mlops.app import app, MODEL_URI


@pytest.fixture(scope="module")
def client():
    # Runs startup, including loading the actual MLflow model.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def payload():
    return {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_uri"] == MODEL_URI


def test_prediction_matches_direct_inference(client, payload):
    # Independently specify the correct feature order.
    expected = int(
        app.state.model.predict([[5.1, 3.5, 1.4, 0.2]])[0]
    )

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": expected,
        "model_uri": MODEL_URI,
    }


@pytest.mark.parametrize("case", ["missing", "negative", "extra"])
def test_invalid_request(client, payload, case):
    if case == "missing":
        del payload["petal width (cm)"]
    elif case == "negative":
        payload["sepal length (cm)"] = -1.0
    else:
        payload["unexpected"] = 1.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 422