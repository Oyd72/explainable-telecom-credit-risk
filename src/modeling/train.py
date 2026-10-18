"""Fit, calibrate and evaluate the selected short-horizon repayment-risk model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from src.modeling.metrics import evaluate_probabilities


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_base_model(config: dict) -> Pipeline:
    params = config["model"]
    classifier = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=int(params["random_state"]),
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", classifier),
        ]
    )


def split_data(df: pd.DataFrame, config: dict):
    date_col = config["data"]["date_column"]
    dates = pd.to_datetime(df[date_col], errors="raise")
    split = config["splits"]

    train = df.loc[dates <= pd.Timestamp(split["training_end"])].copy()
    calibration = df.loc[
        (dates >= pd.Timestamp(split["calibration_start"]))
        & (dates <= pd.Timestamp(split["calibration_end"]))
    ].copy()
    holdout = df.loc[
        (dates >= pd.Timestamp(split["holdout_start"]))
        & (dates <= pd.Timestamp(split["holdout_end"]))
    ].copy()

    if train.empty or calibration.empty or holdout.empty:
        raise ValueError("One or more modelling splits are empty.")
    return train, calibration, holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/repayment_risk_model.joblib"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/tables/model_evaluation.json"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("models/model_metadata.json"),
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    data_path = PROJECT_ROOT / config["data"]["processed_path"]
    df = pd.read_csv(data_path)

    features = config["features"]
    target = config["data"]["target_column"]
    required = set(features + [target, config["data"]["date_column"]])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    train, calibration, holdout = split_data(df, config)

    base_model = build_base_model(config)
    base_model.fit(train[features], train[target])

    calibration_raw = base_model.predict_proba(calibration[features])[:, 1]
    holdout_raw = base_model.predict_proba(holdout[features])[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration_raw, calibration[target].to_numpy())
    holdout_probability = calibrator.predict(holdout_raw)

    threshold = float(config["operating_threshold"])
    metrics = evaluate_probabilities(
        holdout[target].to_numpy(),
        holdout_probability,
        threshold,
    )

    package = {
        "base_model": base_model,
        "calibrator": calibrator,
        "features": features,
        "target": target,
        "operating_threshold": threshold,
        "training_end": config["splits"]["training_end"],
        "calibration_start": config["splits"]["calibration_start"],
        "calibration_end": config["splits"]["calibration_end"],
        "holdout_start": config["splits"]["holdout_start"],
        "holdout_end": config["splits"]["holdout_end"],
    }

    output_path = PROJECT_ROOT / args.output
    metrics_path = PROJECT_ROOT / args.metrics
    metadata_path = PROJECT_ROOT / args.metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(package, output_path)
    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

    result = {
        "training_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "holdout_rows": int(len(holdout)),
        "feature_count": int(len(features)),
        "features": features,
        "threshold": threshold,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    metadata = {
        "model_name": "telecom_repayment_risk_random_forest_isotonic",
        "artifact_filename": output_path.name,
        "artifact_sha256": sha256,
        "feature_count": len(features),
        "features": features,
        "target_definition": "delinquent_5d = 1 when repayment was not completed within five days",
        "training_rows": len(train),
        "calibration_rows": len(calibration),
        "holdout_rows": len(holdout),
        "model_parameters": config["model"],
        "calibration_method": config["calibration"]["method"],
        "operating_threshold": threshold,
        "holdout_used_for_fitting": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Model package written to {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Metadata written to {metadata_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
