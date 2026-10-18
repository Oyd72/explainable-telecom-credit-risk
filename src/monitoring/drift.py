"""Label-free drift metrics for numeric model inputs and prediction scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftThresholds:
    psi_watch: float = 0.10
    psi_escalate: float = 0.25
    missingness_watch_pp: float = 2.0
    missingness_escalate_pp: float = 5.0


def _safe_proportions(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, epsilon, None)
    return values / values.sum()


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate PSI using quantile bins derived from the reference population."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)

    if len(ref) == 0 or len(cur) == 0 or np.unique(ref).size < 2:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return float("nan")

    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p = _safe_proportions(ref_counts)
    cur_p = _safe_proportions(cur_counts)

    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic without scipy."""
    ref = np.sort(pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float))
    cur = np.sort(pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float))
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    grid = np.sort(np.unique(np.concatenate([ref, cur])))
    ref_cdf = np.searchsorted(ref, grid, side="right") / len(ref)
    cur_cdf = np.searchsorted(cur, grid, side="right") / len(cur)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def missingness_change_pp(reference: pd.Series, current: pd.Series) -> float:
    return float(abs(current.isna().mean() - reference.isna().mean()) * 100.0)


def severity_from_thresholds(
    psi: float,
    missingness_pp: float,
    thresholds: DriftThresholds | None = None,
) -> str:
    thresholds = thresholds or DriftThresholds()

    if (
        (np.isfinite(psi) and psi >= thresholds.psi_escalate)
        or missingness_pp >= thresholds.missingness_escalate_pp
    ):
        return "escalate"

    if (
        (np.isfinite(psi) and psi >= thresholds.psi_watch)
        or missingness_pp >= thresholds.missingness_watch_pp
    ):
        return "watch"

    return "ok"


def numeric_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    thresholds: DriftThresholds | None = None,
) -> pd.DataFrame:
    missing = sorted(
        (set(features) - set(reference.columns))
        | (set(features) - set(current.columns))
    )
    if missing:
        raise ValueError(f"Missing monitoring features: {missing}")

    rows = []
    for feature in features:
        psi = population_stability_index(reference[feature], current[feature])
        ks = ks_statistic(reference[feature], current[feature])
        missingness_pp = missingness_change_pp(reference[feature], current[feature])

        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "ks_statistic": ks,
                "reference_missing_rate": float(reference[feature].isna().mean()),
                "current_missing_rate": float(current[feature].isna().mean()),
                "missingness_change_pp": missingness_pp,
                "reference_mean": float(pd.to_numeric(reference[feature], errors="coerce").mean()),
                "current_mean": float(pd.to_numeric(current[feature], errors="coerce").mean()),
                "severity": severity_from_thresholds(psi, missingness_pp, thresholds),
            }
        )

    return pd.DataFrame(rows)


def prediction_drift(
    reference_scores: pd.Series,
    current_scores: pd.Series,
    thresholds: DriftThresholds | None = None,
) -> dict:
    psi = population_stability_index(reference_scores, current_scores)
    ks = ks_statistic(reference_scores, current_scores)
    missingness_pp = missingness_change_pp(reference_scores, current_scores)

    return {
        "psi": psi,
        "ks_statistic": ks,
        "reference_mean": float(pd.to_numeric(reference_scores, errors="coerce").mean()),
        "current_mean": float(pd.to_numeric(current_scores, errors="coerce").mean()),
        "reference_median": float(pd.to_numeric(reference_scores, errors="coerce").median()),
        "current_median": float(pd.to_numeric(current_scores, errors="coerce").median()),
        "missingness_change_pp": missingness_pp,
        "severity": severity_from_thresholds(psi, missingness_pp, thresholds),
    }
