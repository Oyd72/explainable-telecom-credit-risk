import pandas as pd

from src.monitoring.drift import (
    DriftThresholds,
    population_stability_index,
    severity_from_thresholds,
)


def test_population_stability_index_is_near_zero_for_same_distribution():
    reference = pd.Series([1, 2, 3, 4, 5] * 20)
    current = pd.Series([1, 2, 3, 4, 5] * 20)

    psi = population_stability_index(reference, current)

    assert abs(psi) < 1e-12


def test_severity_thresholds():
    thresholds = DriftThresholds(psi_watch=0.10, psi_escalate=0.25)

    assert severity_from_thresholds(0.05, 0.0, thresholds) == "ok"
    assert severity_from_thresholds(0.15, 0.0, thresholds) == "watch"
    assert severity_from_thresholds(0.30, 0.0, thresholds) == "escalate"
