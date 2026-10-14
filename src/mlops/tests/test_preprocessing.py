import numpy as np
import pandas as pd
import pytest

from src.mlops.utils.dataprocessing import prepare_features


FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

@pytest.fixture
def sample():
    return pd.DataFrame({
        "sepal_length": [5.1],
        "sepal_width": [3.5],
        "petal_length": [1.4],
        "petal_width": [0.2],
    })


def test_reorders_columns_without_changing_input(sample):
    shuffled = sample.iloc[:, ::-1].copy()
    original = shuffled.copy(deep=True)

    result = prepare_features(shuffled, FEATURES)

    np.testing.assert_allclose(result, [[5.1, 3.5, 1.4, 0.2]])
    assert result.shape == (1, 4)
    assert result.dtype == np.float64
    pd.testing.assert_frame_equal(shuffled, original)

@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf, 0.0, -1.0, "invalid", True],
)
def test_rejects_invalid_measurements(sample, bad_value):
    bad = sample.assign(sepal_length=bad_value)

    with pytest.raises(ValueError):
        prepare_features(bad, FEATURES)


@pytest.mark.parametrize(
    "case", ["missing", "extra", "duplicate", "empty"]
)
def test_rejects_invalid_schema(sample, case):
    if case == "missing":
        bad = sample.drop(columns="petal_width")
    # elif case == "extra":
    #     bad = sample.assign(unexpected=1.0)
    elif case == "duplicate":
        bad = pd.concat([sample, sample[["sepal_length"]]], axis=1)
    else:
        bad = sample.iloc[:0]

    with pytest.raises(ValueError):
        print("bad", bad)
        prepare_features(bad, FEATURES)