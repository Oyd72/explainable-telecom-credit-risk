"""Run drift and performance monitoring for the reference model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.monitoring.drift import (
    DriftThresholds,
    numeric_feature_drift,
    prediction_drift,
)
from src.monitoring.performance import evaluate_labelled_period


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def score(frame: pd.DataFrame, package: dict) -> pd.Series:
    features = package["features"]
    raw = package["base_model"].predict_proba(frame[features])[:, 1]
    calibrated = package["calibrator"].predict(raw)
    return pd.Series(calibrated, index=frame.index, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/repayment_risk_model.joblib"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/monitoring/reference.csv"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("data/monitoring/comparison_labelled.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/monitoring.yaml"),
    )
    args = parser.parse_args()

    package = joblib.load(PROJECT_ROOT / args.model)
    reference = pd.read_csv(PROJECT_ROOT / args.reference)
    comparison = pd.read_csv(PROJECT_ROOT / args.comparison)
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))

    thresholds = DriftThresholds(
        psi_watch=float(config["drift"]["psi_watch"]),
        psi_escalate=float(config["drift"]["psi_escalate"]),
        missingness_watch_pp=float(config["drift"]["missingness_watch_pp"]),
        missingness_escalate_pp=float(config["drift"]["missingness_escalate_pp"]),
    )

    features = list(package["features"])
    reference_scores = score(reference, package)
    comparison_scores = score(comparison, package)

    feature_drift = numeric_feature_drift(
        reference,
        comparison,
        features,
        thresholds,
    )
    score_drift = prediction_drift(
        reference_scores,
        comparison_scores,
        thresholds,
    )

    threshold = float(config["performance"]["operating_threshold"])
    performance = evaluate_labelled_period(
        comparison["delinquent_5d"],
        comparison_scores,
        threshold,
        period_name="2016-07-14_to_2016-07-23",
        labels_reliable=True,
    )

    severities = set(feature_drift["severity"].astype(str))
    severities.add(str(score_drift["severity"]))
    overall = (
        "escalate"
        if "escalate" in severities
        else "watch"
        if "watch" in severities
        else "ok"
    )

    status = {
        "model_name": "telecom_repayment_risk_random_forest_isotonic",
        "reference_rows": int(len(reference)),
        "comparison_rows": int(len(comparison)),
        "overall_drift_status": overall,
        "features_escalate": feature_drift.loc[
            feature_drift["severity"] == "escalate", "feature"
        ].tolist(),
        "features_watch": feature_drift.loc[
            feature_drift["severity"] == "watch", "feature"
        ].tolist(),
        "prediction_drift": score_drift,
        "comparison_performance": performance,
        "fairness_boundary": (
            "The dataset has no reliable protected demographic attributes. "
            "Operational drift must not be presented as demographic fairness evidence."
        ),
    }

    output_dir = PROJECT_ROOT / "reports/monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_drift.to_csv(output_dir / "feature_drift.csv", index=False)
    (output_dir / "latest_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
