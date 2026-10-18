"""Outcome-based model monitoring for periods with trustworthy labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling.metrics import evaluate_probabilities


def evaluate_labelled_period(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    *,
    period_name: str,
    labels_reliable: bool,
) -> dict:
    if not labels_reliable:
        raise ValueError(
            "Outcome-based monitoring requires reliable matured labels. "
            "Use label-free drift monitoring until such outcomes are available."
        )

    metrics = evaluate_probabilities(
        pd.Series(y_true).astype(int).to_numpy(),
        pd.Series(probabilities).astype(float).to_numpy(),
        threshold,
    )
    return {
        "period_name": period_name,
        "rows": int(len(y_true)),
        **metrics,
        "operating_threshold": float(threshold),
        "labels_reliable": True,
    }
