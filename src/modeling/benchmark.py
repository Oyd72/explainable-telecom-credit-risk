"""Compare baseline and nonlinear candidate models on expanding chronological folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.modeling.metrics import top_fraction_capture


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOLDS = [
    {"name": "late_june", "train_end": "2016-06-13", "eval_start": "2016-06-14", "eval_end": "2016-06-23"},
    {"name": "turn_of_month", "train_end": "2016-06-23", "eval_start": "2016-06-24", "eval_end": "2016-07-03"},
    {"name": "early_july", "train_end": "2016-07-03", "eval_start": "2016-07-04", "eval_end": "2016-07-13"},
]


def evaluate(model, x_train, y_train, x_eval, y_eval) -> dict[str, float]:
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_eval)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_eval, probability)),
        "average_precision": float(average_precision_score(y_eval, probability)),
        "brier": float(brier_score_loss(y_eval, probability)),
        "top20_capture": top_fraction_capture(y_eval.to_numpy(), probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/model_benchmark.json"),
    )
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    df = pd.read_csv(PROJECT_ROOT / config["data"]["processed_path"])
    date_col = config["data"]["date_column"]
    target = config["data"]["target_column"]
    features = config["features"]
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    development = df.loc[df[date_col] <= pd.Timestamp("2016-07-13")].copy()

    numeric_scaled = ColumnTransformer(
        [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features)],
        remainder="drop",
    )
    numeric = ColumnTransformer(
        [("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), features)],
        remainder="drop",
    )

    records = []
    for fold in FOLDS:
        train = development.loc[development[date_col] <= pd.Timestamp(fold["train_end"])]
        evaluation = development.loc[
            (development[date_col] >= pd.Timestamp(fold["eval_start"]))
            & (development[date_col] <= pd.Timestamp(fold["eval_end"]))
        ]
        y_train = train[target]
        models = {
            "logistic_regression": Pipeline([
                ("prep", numeric_scaled),
                ("model", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)),
            ]),
            "random_forest": Pipeline([
                ("prep", numeric),
                ("model", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
            ]),
            "xgboost": Pipeline([
                ("prep", numeric),
                ("model", XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
                    reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
                    scale_pos_weight=int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1),
                    n_jobs=-1, random_state=42,
                )),
            ]),
        }

        for name, model in models.items():
            metrics = evaluate(
                model,
                train[features],
                y_train,
                evaluation[features],
                evaluation[target],
            )
            records.append({"model": name, "fold": fold["name"], **metrics})

    result = pd.DataFrame(records)
    summary = (
        result.groupby("model", as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            min_roc_auc=("roc_auc", "min"),
            mean_average_precision=("average_precision", "mean"),
            mean_brier=("brier", "mean"),
            mean_top20_capture=("top20_capture", "mean"),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )

    payload = {
        "holdout_opened": False,
        "folds": FOLDS,
        "feature_count": len(features),
        "fold_metrics": result.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Benchmark written to {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
