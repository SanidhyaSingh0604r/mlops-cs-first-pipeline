
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    """
    Verify that the app is running and can respond to a health-check request.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_iris_prediction():
    """
    Verify that we are getting valid predictions.
    """
    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "class-name" in result
    assert result["class-name"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]
    assert result["prediction"] in [0, 1, 2]
