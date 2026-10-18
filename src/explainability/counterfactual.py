"""Generate constrained counterfactual-style explanations for selected cases.

The analysis asks whether small changes drawn from the observed training
distribution would move a model score across the operating threshold. It is a
model-behaviour explanation, not a causal claim or a recommendation to customers.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models/repayment_risk_model.joblib"
DEFAULT_DATA = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIGURE_DIR = PROJECT_ROOT / "reports/figures/explainability"

SINGLE_QUANTILES = np.linspace(0.01, 0.99, 41)
PAIR_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def calibrated_scores(frame, package) -> np.ndarray:
    raw = package["base_model"].predict_proba(frame[package["features"]])[:, 1]
    return np.asarray(package["calibrator"].predict(raw), dtype=float)


def observed_grid(series: pd.Series, quantiles) -> list[float]:
    values = series.dropna().to_numpy(dtype=float)
    unique = np.unique(values)
    if len(unique) == 0:
        return []
    targets = np.quantile(values, quantiles)
    return sorted(set(float(unique[np.abs(unique - value).argmin()]) for value in targets))


def iqr(series: pd.Series) -> float:
    return max(float(series.quantile(0.75) - series.quantile(0.25)), 1e-9)


def choose_cases(holdout: pd.DataFrame, scores: np.ndarray, threshold: float) -> list[dict]:
    series = pd.Series(scores, index=holdout.index)
    below = series.loc[series < threshold]
    above = series.loc[series >= threshold]
    if below.empty or above.empty:
        raise ValueError("Holdout does not contain cases on both sides of the threshold.")

    below_index = (threshold - below).idxmin()
    above_index = (above - threshold).idxmin()
    q90 = float(series.quantile(0.90))
    high_index = (series - q90).abs().idxmin()

    return [
        {"case": "near_boundary_lower_risk", "index": int(below_index), "direction": "up"},
        {"case": "near_boundary_higher_risk", "index": int(above_index), "direction": "down"},
        {"case": "representative_high_risk", "index": int(high_index), "direction": "down"},
    ]


def crossed(score: float, threshold: float, direction: str) -> bool:
    return score >= threshold if direction == "up" else score < threshold


def single_feature_search(base_row, direction, threshold, train, package):
    candidates = []
    features = package["features"]

    for feature in features:
        original = float(base_row.iloc[0][feature])
        scale = iqr(train[feature])
        rows, values = [], []

        for value in observed_grid(train[feature], SINGLE_QUANTILES):
            if np.isclose(value, original, equal_nan=True):
                continue
            row = base_row.copy()
            row.loc[row.index[0], feature] = value
            rows.append(row.iloc[0][features].to_dict())
            values.append(value)

        if not rows:
            continue

        frame = pd.DataFrame(rows)
        scores = calibrated_scores(frame, package)

        for value, score in zip(values, scores):
            if crossed(float(score), threshold, direction):
                candidates.append(
                    {
                        "search_type": "single_feature",
                        "changes": [{
                            "feature": feature,
                            "from": original,
                            "to": float(value),
                            "delta": float(value - original),
                            "normalized_abs_change_iqr": float(abs(value - original) / scale),
                        }],
                        "counterfactual_score": float(score),
                        "distance": float(abs(value - original) / scale),
                    }
                )

    if not candidates:
        return None
    return min(candidates, key=lambda item: (item["distance"], abs(item["counterfactual_score"] - threshold)))


def two_feature_search(base_row, direction, threshold, train, package, top_features):
    candidates = []
    features = package["features"]

    for a, b in combinations(top_features, 2):
        original_a = float(base_row.iloc[0][a])
        original_b = float(base_row.iloc[0][b])
        scale_a, scale_b = iqr(train[a]), iqr(train[b])

        rows, changes = [], []
        for value_a, value_b in product(
            observed_grid(train[a], PAIR_QUANTILES),
            observed_grid(train[b], PAIR_QUANTILES),
        ):
            if np.isclose(value_a, original_a) and np.isclose(value_b, original_b):
                continue
            row = base_row.copy()
            row.loc[row.index[0], a] = value_a
            row.loc[row.index[0], b] = value_b
            rows.append(row.iloc[0][features].to_dict())
            distance = abs(value_a - original_a) / scale_a + abs(value_b - original_b) / scale_b
            changes.append((value_a, value_b, float(distance)))

        if not rows:
            continue

        scores = calibrated_scores(pd.DataFrame(rows), package)
        for (value_a, value_b, distance), score in zip(changes, scores):
            if crossed(float(score), threshold, direction):
                candidates.append(
                    {
                        "search_type": "two_feature",
                        "changes": [
                            {
                                "feature": a,
                                "from": original_a,
                                "to": float(value_a),
                                "delta": float(value_a - original_a),
                                "normalized_abs_change_iqr": float(abs(value_a - original_a) / scale_a),
                            },
                            {
                                "feature": b,
                                "from": original_b,
                                "to": float(value_b),
                                "delta": float(value_b - original_b),
                                "normalized_abs_change_iqr": float(abs(value_b - original_b) / scale_b),
                            },
                        ],
                        "counterfactual_score": float(score),
                        "distance": distance,
                    }
                )

    if not candidates:
        return None
    return min(candidates, key=lambda item: (item["distance"], abs(item["counterfactual_score"] - threshold)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--shap-importance",
        type=Path,
        default=TABLE_DIR / "shap_global_importance.csv",
    )
    args = parser.parse_args()

    package = joblib.load(args.model)
    threshold = float(package["operating_threshold"])

    df = pd.read_csv(args.data)
    df["pdate"] = pd.to_datetime(df["pdate"], errors="raise")
    train = df.loc[df["pdate"] <= pd.Timestamp(package["training_end"])].copy()
    holdout = df.loc[
        (df["pdate"] >= pd.Timestamp(package["holdout_start"]))
        & (df["pdate"] <= pd.Timestamp(package["holdout_end"]))
    ].copy()

    scores = calibrated_scores(holdout, package)
    cases = choose_cases(holdout, scores, threshold)

    if args.shap_importance.exists():
        importance = pd.read_csv(args.shap_importance)
        top_features = importance.head(5)["feature"].tolist()
    else:
        top_features = list(package["features"][:5])

    summaries = []
    rows = []

    for case in cases:
        index = case["index"]
        base_row = holdout.loc[[index]].copy()
        original_score = float(scores[holdout.index.get_loc(index)])

        result = single_feature_search(
            base_row, case["direction"], threshold, train, package
        )
        if result is None:
            result = two_feature_search(
                base_row, case["direction"], threshold, train, package, top_features
            )

        summary = {
            "case": case["case"],
            "holdout_index": index,
            "observed_outcome": int(base_row.iloc[0]["delinquent_5d"]),
            "original_score": original_score,
            "threshold": threshold,
            "counterfactual_found": result is not None,
        }

        if result is not None:
            summary.update(
                {
                    "search_type": result["search_type"],
                    "counterfactual_score": result["counterfactual_score"],
                    "normalized_distance": result["distance"],
                    "changes": result["changes"],
                }
            )
            for change in result["changes"]:
                rows.append(
                    {
                        "case": case["case"],
                        "original_score": original_score,
                        "counterfactual_score": result["counterfactual_score"],
                        "threshold": threshold,
                        "search_type": result["search_type"],
                        **change,
                    }
                )

        summaries.append(summary)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(TABLE_DIR / "counterfactual_explanations.csv", index=False)
    payload = {
        "threshold": threshold,
        "cases": summaries,
        "interpretation_caution": (
            "Counterfactuals describe the fitted model's response to constrained input changes. "
            "They are not causal claims and are not recommendations for customer behaviour."
        ),
    }
    (TABLE_DIR / "counterfactual_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    chart_rows = [
        {
            "case": item["case"],
            "original": item["original_score"],
            "counterfactual": item.get("counterfactual_score", np.nan),
        }
        for item in summaries
    ]
    chart = pd.DataFrame(chart_rows)
    ax = chart.set_index("case")[["original", "counterfactual"]].plot(
        kind="bar", figsize=(9, 5)
    )
    ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.set_ylabel("Calibrated delinquency risk")
    ax.set_xlabel("")
    ax.set_title("Constrained counterfactual score changes")
    ax.tick_params(axis="x", rotation=15)
    ax.figure.tight_layout()
    ax.figure.savefig(
        FIGURE_DIR / "counterfactual_risk_changes.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(ax.figure)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
