import mlflow.sklearn
import numpy as np
import pytest

# Replace with your actual model name and fixed version.
MODEL_URI = "models:/iris_model@production"

@pytest.fixture(scope="module")
def model():
    return mlflow.sklearn.load_model(MODEL_URI)

def test_known_sample_prediction(model):
    # A known Iris setosa sample: four features, class 0.
    inputs = np.array([[5.1, 3.5, 1.4, 0.2]])

    predictions = model.predict(inputs)

    assert predictions.shape == (1,)
    assert predictions[0] == 0

def test_wrong_feature_count_is_rejected(model):
    inputs = np.array([[5.1, 3.5, 1.4]])  # Missing one feature.

    with pytest.raises(ValueError):
        model.predict(inputs)