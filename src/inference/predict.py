"""Batch inference for the packaged repayment-risk model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models/repayment_risk_model.joblib"


def validate_input(frame: pd.DataFrame, required_features: list[str]) -> None:
    missing = sorted(set(required_features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate input columns are not allowed: {duplicates}")


def predict_frame(frame: pd.DataFrame, package: dict, keep_extra_columns: bool = False) -> pd.DataFrame:
    features = list(package["features"])
    validate_input(frame, features)

    raw = np.asarray(package["base_model"].predict_proba(frame[features])[:, 1], dtype=float)
    calibrated = np.asarray(package["calibrator"].predict(raw), dtype=float)

    output = frame.copy() if keep_extra_columns else pd.DataFrame(index=frame.index)
    output["raw_delinquency_probability"] = raw
    output["calibrated_delinquency_probability"] = calibrated
    output["higher_risk_flag"] = calibrated >= float(package["operating_threshold"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--keep-extra-columns", action="store_true")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model artefact not found: {args.model}. Run python -m src.modeling.train first."
        )

    package = joblib.load(args.model)
    frame = pd.read_csv(args.input)
    result = predict_frame(frame, package, args.keep_extra_columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Rows scored: {len(result):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
