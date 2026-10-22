import numpy as np

from src.modeling.metrics import (
    expected_calibration_error,
    top_fraction_capture,
)


def test_top_fraction_capture_identifies_highest_risk_positive():
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

    capture = top_fraction_capture(y, p, fraction=0.2)

    assert capture == 0.5


def test_expected_calibration_error_is_zero_for_perfect_bins():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(y, p, bins=2) == 0.0
