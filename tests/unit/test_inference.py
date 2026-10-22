import pandas as pd
import pytest

from src.inference.predict import validate_input


FEATURES = ["a", "b", "c"]


def test_validate_input_accepts_required_features():
    frame = pd.DataFrame([{"a": 1, "b": 2, "c": 3}])

    validate_input(frame, FEATURES)


def test_validate_input_rejects_missing_feature():
    frame = pd.DataFrame([{"a": 1, "b": 2}])

    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_input(frame, FEATURES)


def test_validate_input_rejects_duplicate_columns():
    frame = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "b"])

    with pytest.raises(ValueError, match="Duplicate input columns"):
        validate_input(frame, ["a", "b"])
