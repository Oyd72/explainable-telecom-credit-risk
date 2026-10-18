"""Optional Evidently drift cross-check for the same monitoring populations."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def score(frame: pd.DataFrame, package: dict) -> pd.Series:
    features = package["features"]
    raw = package["base_model"].predict_proba(frame[features])[:, 1]
    return pd.Series(package["calibrator"].predict(raw), index=frame.index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/repayment_risk_model.joblib"))
    parser.add_argument("--reference", type=Path, default=Path("data/monitoring/reference.csv"))
    parser.add_argument("--comparison", type=Path, default=Path("data/monitoring/comparison_labelled.csv"))
    args = parser.parse_args()

    package = joblib.load(PROJECT_ROOT / args.model)
    reference = pd.read_csv(PROJECT_ROOT / args.reference)
    comparison = pd.read_csv(PROJECT_ROOT / args.comparison)
    features = package["features"]

    reference_eval = reference[features].copy()
    comparison_eval = comparison[features].copy()
    reference_eval["calibrated_delinquency_probability"] = score(reference, package).to_numpy()
    comparison_eval["calibrated_delinquency_probability"] = score(comparison, package).to_numpy()

    report = Report([DataDriftPreset(method="psi")], include_tests=True)
    snapshot = report.run(comparison_eval, reference_eval)

    output_dir = PROJECT_ROOT / "reports/monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_dir / "evidently_drift.html"))
    (output_dir / "evidently_drift.json").write_text(snapshot.json(), encoding="utf-8")

    print("Evidently drift cross-check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
