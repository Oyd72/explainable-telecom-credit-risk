"""Create and evaluate a governed retraining candidate.

A later labelled batch is split chronologically into an update portion and a
validation portion. Passing the configured gates creates a candidate artefact
only. It never replaces the reference model automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.modeling.metrics import top_fraction_capture


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-data",
        type=Path,
        default=Path("data/processed/telecom_delinquency_model_ready.csv"),
    )
    parser.add_argument(
        "--new-data",
        type=Path,
        default=Path("data/monitoring/comparison_labelled.csv"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model.yaml"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/retraining.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/candidates/retrained_candidate.joblib"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/monitoring/retraining_candidate.json"),
    )
    args = parser.parse_args()

    model_config = yaml.safe_load(
        (PROJECT_ROOT / args.model_config).read_text(encoding="utf-8")
    )
    policy = yaml.safe_load(
        (PROJECT_ROOT / args.policy).read_text(encoding="utf-8")
    )["retraining"]

    features = list(model_config["features"])
    target = model_config["data"]["target_column"]
    date_column = model_config["data"]["date_column"]

    base = pd.read_csv(PROJECT_ROOT / args.base_data)
    new = pd.read_csv(PROJECT_ROOT / args.new_data)

    required = set(features + [target, date_column])
    for name, frame in [("base", base), ("new", new)]:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} data are missing required columns: {missing}")

    base[date_column] = pd.to_datetime(base[date_column], errors="raise")
    new[date_column] = pd.to_datetime(new[date_column], errors="raise")
    new = new.sort_values(date_column).reset_index(drop=True)

    minimum_rows = int(policy["trigger"]["minimum_new_labelled_rows"])
    if len(new) < minimum_rows:
        raise ValueError(
            f"New labelled batch has {len(new):,} rows; "
            f"at least {minimum_rows:,} are required."
        )

    validation_fraction = float(policy["evaluation"]["validation_fraction"])
    split_index = int(np.floor(len(new) * (1.0 - validation_fraction)))
    if split_index <= 0 or split_index >= len(new):
        raise ValueError("Chronological validation split is empty.")

    update_batch = new.iloc[:split_index].copy()
    validation = new.iloc[split_index:].copy()

    original_development_end = pd.Timestamp("2016-07-13")
    historical = base.loc[base[date_column] <= original_development_end].copy()
    combined = (
        pd.concat([historical, update_batch], ignore_index=True)
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    unique_dates = sorted(combined[date_column].dt.normalize().unique())
    if len(unique_dates) < 8:
        raise ValueError("Not enough dated observations for a seven-day calibration slice.")

    calibration_start = pd.Timestamp(unique_dates[-7])
    train = combined.loc[combined[date_column] < calibration_start].copy()
    calibration = combined.loc[combined[date_column] >= calibration_start].copy()

    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    x_calibration = imputer.transform(calibration[features])
    x_validation = imputer.transform(validation[features])

    params = model_config["model"]
    model = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=int(policy["evaluation"]["random_state"]),
        n_jobs=-1,
    )
    model.fit(x_train, train[target])

    raw_calibration = model.predict_proba(x_calibration)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_calibration, calibration[target].to_numpy())

    raw_validation = model.predict_proba(x_validation)[:, 1]
    calibrated_validation = calibrator.predict(raw_validation)

    metrics = {
        "roc_auc": float(roc_auc_score(validation[target], calibrated_validation)),
        "average_precision": float(
            average_precision_score(validation[target], calibrated_validation)
        ),
        "brier_score": float(
            brier_score_loss(validation[target], calibrated_validation)
        ),
        "top20_capture": top_fraction_capture(
            validation[target].to_numpy(),
            calibrated_validation,
        ),
    }

    gates = {
        "roc_auc": (
            metrics["roc_auc"]
            >= float(policy["evaluation"]["minimum_roc_auc"])
        ),
        "top20_capture": (
            metrics["top20_capture"]
            >= float(policy["evaluation"]["minimum_top20_capture"])
        ),
        "brier_score": (
            metrics["brier_score"]
            <= float(policy["evaluation"]["maximum_brier_score"])
        ),
    }
    passed = all(gates.values())

    package = {
        "candidate_only": True,
        "automatic_promotion": False,
        "features": features,
        "target": target,
        "imputer": imputer,
        "model": model,
        "calibrator": calibrator,
        "training_end": str(train[date_column].max().date()),
        "calibration_start": str(calibration[date_column].min().date()),
        "calibration_end": str(calibration[date_column].max().date()),
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, output_path)

    summary = {
        "candidate_created": True,
        "candidate_only": True,
        "automatic_promotion": False,
        "human_approval_required": True,
        "new_labelled_rows": int(len(new)),
        "update_rows": int(len(update_batch)),
        "validation_rows": int(len(validation)),
        "validation_period": {
            "start": str(validation[date_column].min().date()),
            "end": str(validation[date_column].max().date()),
        },
        "metrics": metrics,
        "quality_gates": gates,
        "all_quality_gates_passed": passed,
        "candidate_sha256": sha256_file(output_path),
        "promotion_decision": (
            "Candidate only. Promotion requires explicit human review."
        ),
    }

    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
