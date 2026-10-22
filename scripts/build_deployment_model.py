"""Build the deployable model artefact from the public Kaggle dataset.

This script is intended for deployment environments such as Render. It downloads
the public source data at build time, applies the same cleaning and model-ready
transformation logic used by the repository, trains the reference model and
writes only the generated model artefact.

Raw and derived row-level data are deleted before the build step exits.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import kagglehub
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline

from src.data.clean import clean_dataframe
from src.features.build_dataset import build_model_dataset


DATA_HANDLE = "sivakrishna3311/delinquency-telecom-dataset"
SOURCE_FILENAME = "sample_data_intw.csv"


def locate_source(downloaded: str | Path) -> Path:
    path = Path(downloaded)
    if path.is_file():
        return path

    exact = path / SOURCE_FILENAME
    if exact.exists():
        return exact

    matches = list(path.rglob(SOURCE_FILENAME))
    if not matches:
        raise FileNotFoundError(
            f"{SOURCE_FILENAME} was not found in the downloaded Kaggle dataset."
        )
    return matches[0]


def main() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "config/model.yaml").read_text(encoding="utf-8")
    )

    temp_dir = PROJECT_ROOT / ".deployment_data"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading public Kaggle dataset...")
    downloaded = kagglehub.dataset_download(
        DATA_HANDLE,
        path=SOURCE_FILENAME,
        output_dir=str(temp_dir),
    )
    source = locate_source(downloaded)

    try:
        print("Preparing model-ready data...")
        raw = pd.read_csv(source)
        now = datetime.now(timezone.utc)
        cleaned, _, _ = clean_dataframe(
            raw,
            run_id=now.strftime("deploy_%Y%m%dT%H%M%S_%fZ"),
            run_timestamp=now.isoformat(),
        )
        model_ready, summary = build_model_dataset(cleaned)

        features = list(config["features"])
        target = config["data"]["target_column"]
        date_column = config["data"]["date_column"]
        model_ready[date_column] = pd.to_datetime(
            model_ready[date_column],
            errors="raise",
        )

        split = config["splits"]
        train = model_ready.loc[
            model_ready[date_column] <= pd.Timestamp(split["training_end"])
        ].copy()
        calibration = model_ready.loc[
            (model_ready[date_column] >= pd.Timestamp(split["calibration_start"]))
            & (model_ready[date_column] <= pd.Timestamp(split["calibration_end"]))
        ].copy()

        params = config["model"]
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train[features])
        x_calibration = imputer.transform(calibration[features])

        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=int(params["random_state"]),
            n_jobs=-1,
        )
        model.fit(x_train, train[target])

        raw_calibration = model.predict_proba(x_calibration)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(
            raw_calibration,
            calibration[target].to_numpy(),
        )

        pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("model", model),
            ]
        )

        package = {
            "base_model": pipeline,
            "calibrator": calibrator,
            "features": features,
            "target": target,
            "operating_threshold": float(config["operating_threshold"]),
            "training_end": split["training_end"],
            "calibration_start": split["calibration_start"],
            "calibration_end": split["calibration_end"],
            "holdout_start": split["holdout_start"],
            "holdout_end": split["holdout_end"],
        }

        model_path = PROJECT_ROOT / "models/repayment_risk_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(package, model_path, compress=3)

        print(
            f"Deployment model written to {model_path}. "
            f"Model-ready rows used for the validated population: {summary['output_rows']:,}."
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
