"""Generate global and local SHAP explanations for the reference model.

SHAP is applied to the fitted Random Forest before isotonic calibration. The
calibrator is a monotonic post-processing layer that changes the probability
scale but not the tree model's feature logic. Local outputs therefore report
both the raw Random Forest probability and the calibrated risk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models/repayment_risk_model.joblib"
DEFAULT_DATA = PROJECT_ROOT / "data/processed/telecom_delinquency_model_ready.csv"
TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIGURE_DIR = PROJECT_ROOT / "reports/figures/explainability"

GLOBAL_SAMPLE_SIZE = 2_000
RANDOM_STATE = 42


def positive_class_shap_values(explainer, x: pd.DataFrame) -> tuple[np.ndarray, float]:
    values = explainer.shap_values(x)

    if isinstance(values, list):
        positive = np.asarray(values[1])
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            positive = values[:, :, 1]
        elif values.ndim == 2:
            positive = values
        else:
            raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        base_value = float(np.asarray(expected).reshape(-1)[-1])
    else:
        base_value = float(expected)
    return positive, base_value


def choose_representative_cases(holdout: pd.DataFrame, scores: np.ndarray) -> list[dict]:
    series = pd.Series(scores, index=holdout.index)
    result = []
    for name, quantile in [("low_risk", 0.10), ("typical_risk", 0.50), ("high_risk", 0.90)]:
        target = float(series.quantile(quantile))
        index = (series - target).abs().idxmin()
        result.append(
            {
                "case": name,
                "quantile_target": quantile,
                "index": int(index),
                "calibrated_risk": float(series.loc[index]),
                "observed_outcome": int(holdout.loc[index, "delinquent_5d"]),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    package = joblib.load(args.model)
    features = list(package["features"])
    base_pipeline = package["base_model"]
    calibrator = package["calibrator"]

    df = pd.read_csv(args.data)
    df["pdate"] = pd.to_datetime(df["pdate"], errors="raise")
    holdout = df.loc[
        (df["pdate"] >= pd.Timestamp(package["holdout_start"]))
        & (df["pdate"] <= pd.Timestamp(package["holdout_end"]))
    ].copy()

    if holdout.empty:
        raise ValueError("Holdout population is empty.")

    imputer = base_pipeline.named_steps["imputer"]
    model = base_pipeline.named_steps["model"]

    x_holdout = pd.DataFrame(
        imputer.transform(holdout[features]),
        columns=features,
        index=holdout.index,
    )
    raw_scores = model.predict_proba(x_holdout)[:, 1]
    calibrated_scores = np.asarray(calibrator.predict(raw_scores), dtype=float)

    rng = np.random.default_rng(RANDOM_STATE)
    if len(x_holdout) > GLOBAL_SAMPLE_SIZE:
        positions = np.sort(rng.choice(len(x_holdout), size=GLOBAL_SAMPLE_SIZE, replace=False))
        x_global = x_holdout.iloc[positions].copy()
    else:
        x_global = x_holdout.copy()

    explainer = shap.TreeExplainer(model)
    global_shap, base_value = positive_class_shap_values(explainer, x_global)

    global_importance = (
        pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": np.abs(global_shap).mean(axis=0),
                "mean_shap": global_shap.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    global_importance.to_csv(TABLE_DIR / "shap_global_importance.csv", index=False)

    plot_df = global_importance.sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    ax.set_title("Global SHAP importance")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_global_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    explanation = shap.Explanation(
        values=global_shap,
        base_values=np.full(len(x_global), base_value),
        data=x_global.to_numpy(),
        feature_names=features,
    )
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=len(features), show=False)
    plt.title("SHAP distribution on holdout sample")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_global_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    cases = choose_representative_cases(holdout, calibrated_scores)
    local_rows = []
    case_summaries = []

    for case in cases:
        index = case["index"]
        x_case = x_holdout.loc[[index]]
        shap_values, local_base = positive_class_shap_values(explainer, x_case)
        raw_risk = float(model.predict_proba(x_case)[:, 1][0])

        contributions = (
            pd.DataFrame(
                {
                    "feature": features,
                    "feature_value": x_case.iloc[0].to_numpy(),
                    "shap_value": shap_values[0],
                    "abs_shap": np.abs(shap_values[0]),
                }
            )
            .sort_values("abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        for rank, row in enumerate(contributions.itertuples(index=False), start=1):
            local_rows.append(
                {
                    "case": case["case"],
                    "holdout_index": index,
                    "observed_outcome": case["observed_outcome"],
                    "raw_risk": raw_risk,
                    "calibrated_risk": case["calibrated_risk"],
                    "base_value": local_base,
                    "importance_rank": rank,
                    "feature": row.feature,
                    "feature_value": float(row.feature_value),
                    "shap_value": float(row.shap_value),
                    "abs_shap": float(row.abs_shap),
                    "direction": (
                        "increases predicted delinquency risk"
                        if row.shap_value > 0
                        else "decreases predicted delinquency risk"
                        if row.shap_value < 0
                        else "neutral"
                    ),
                }
            )

        local_explanation = shap.Explanation(
            values=shap_values[0],
            base_values=local_base,
            data=x_case.iloc[0].to_numpy(),
            feature_names=features,
        )
        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(local_explanation, max_display=len(features), show=False)
        plt.title(
            f"{case['case'].replace('_', ' ').title()} — "
            f"raw={raw_risk:.1%}, calibrated={case['calibrated_risk']:.1%}"
        )
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / f"shap_local_{case['case']}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

        case_summaries.append(
            {
                **case,
                "raw_risk": raw_risk,
                "top_contributors": contributions.head(5)[
                    ["feature", "feature_value", "shap_value"]
                ].to_dict(orient="records"),
            }
        )

    pd.DataFrame(local_rows).to_csv(TABLE_DIR / "shap_local_cases.csv", index=False)

    summary = {
        "shap_explains": "base_random_forest_positive_class_output",
        "calibration": "isotonic",
        "global_sample_rows": int(len(x_global)),
        "top_global_features": global_importance.head(10).to_dict(orient="records"),
        "representative_cases": case_summaries,
        "interpretation_caution": (
            "SHAP describes how the fitted model uses features. It does not establish "
            "causal effects or show that changing a feature would change repayment behaviour."
        ),
    }
    (TABLE_DIR / "shap_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(global_importance.head(10).to_string(index=False))
    print("SHAP outputs written under reports/tables and reports/figures/explainability.")


if __name__ == "__main__":
    main()
