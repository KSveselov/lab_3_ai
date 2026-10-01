import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from e1 import train_and_predict
from split_data import load_data


@pytest.fixture(scope="module")
def trained_model():
    model, _, _, _ = train_and_predict()
    return model


def test_train_and_test_indices_do_not_intersect():
    train_df, test_df = load_data()

    assert set(train_df.index).isdisjoint(test_df.index)


def test_pipeline_accepts_unknown_species(trained_model):
    _, test_df = load_data()
    features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    unknown_fish = test_df.loc[:, features].head(1).copy()
    unknown_fish["Species"] = "UnknownFish"

    prediction = trained_model.predict(unknown_fish)

    assert len(prediction) == 1
    assert np.isfinite(prediction).all()


def test_predictions_match_input_size_and_are_finite(trained_model):
    _, test_df = load_data()
    features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    prediction = trained_model.predict(test_df.loc[:, features])

    assert len(prediction) == len(test_df)
    assert np.isfinite(prediction).all()


def test_metrics_match_manual_calculation():
    actual = [100.0, 200.0]
    prediction = [110.0, 180.0]

    assert mean_absolute_error(actual, prediction) == pytest.approx(15.0)
    assert mean_squared_error(actual, prediction) == pytest.approx(250.0)


def test_saved_pipeline_reproduces_predictions(trained_model, tmp_path):
    _, test_df = load_data()
    features = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    input_data = test_df.loc[:, features].head(3)
    expected_prediction = trained_model.predict(input_data)
    model_path = tmp_path / "fish_model.joblib"

    joblib.dump(trained_model, model_path)
    loaded_model = joblib.load(model_path)

    np.testing.assert_allclose(
        loaded_model.predict(input_data), expected_prediction, rtol=1e-8, atol=1e-8
    )
