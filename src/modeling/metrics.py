"""Evaluation metrics used by the repayment-risk modelling workflow."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def top_fraction_capture(y_true, probabilities, fraction: float = 0.20) -> float:
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    positives = y.sum()
    if positives == 0:
        return float("nan")
    n = max(1, int(np.ceil(len(p) * fraction)))
    order = np.argsort(-p)[:n]
    return float(y[order].sum() / positives)


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index < bins - 1:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        if mask.any():
            error += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def evaluate_probabilities(y_true, probabilities, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    predicted = (p >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": expected_calibration_error(y, p, bins=10),
        "top20_capture": top_fraction_capture(y, p, fraction=0.20),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "mean_predicted_risk": float(p.mean()),
        "observed_delinquency_rate": float(y.mean()),
    }
