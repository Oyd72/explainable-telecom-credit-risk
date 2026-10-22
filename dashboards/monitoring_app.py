"""Streamlit dashboard for drift, performance and governance evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "reports/monitoring/latest_status.json"
DRIFT = ROOT / "reports/monitoring/feature_drift.csv"
POPULATION = ROOT / "reports/monitoring/population_summary.json"
EVIDENTLY = ROOT / "reports/monitoring/evidently_reference.json"

DRIFT_THRESHOLD = 0.10
PREDICTION_COLUMN = "calibrated_delinquency_probability"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_evidently(payload: dict) -> pd.DataFrame:
    rows = []
    for metric in payload.get("metrics", []):
        config = metric.get("config") or {}
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        value = metric.get("value")
        column = config.get("column")
        if column is None or not isinstance(value, (int, float)):
            continue
        threshold = float(config.get("threshold", DRIFT_THRESHOLD))
        rows.append(
            {
                "column": str(column),
                "evidently_psi": float(value),
                "evidently_threshold": threshold,
                "evidently_drift": float(value) >= threshold,
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Credit-risk monitoring", page_icon="📈", layout="wide")
st.title("Telecom delinquency model monitoring")
st.caption(
    "Governance-facing monitoring for distribution drift, predictive performance and evidence limits."
)

required = [STATUS, DRIFT, POPULATION, EVIDENTLY]
missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
if missing:
    st.error("Missing monitoring evidence: " + ", ".join(missing))
    st.stop()

status = read_json(STATUS)
drift = pd.read_csv(DRIFT)
population = read_json(POPULATION)
evidently = extract_evidently(read_json(EVIDENTLY))

prediction_row = pd.DataFrame(
    [
        {
            "feature": PREDICTION_COLUMN,
            "psi": float(status["prediction_drift"]["psi"]),
            "ks_statistic": float(status["prediction_drift"]["ks_statistic"]),
            "severity": str(status["prediction_drift"]["severity"]),
        }
    ]
)
project_all = pd.concat([drift, prediction_row], ignore_index=True)

tabs = st.tabs(
    ["Model health", "Data & prediction drift", "Performance & calibration", "Fairness & robustness"]
)

with tabs[0]:
    st.subheader("Model and monitoring status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Random Forest + isotonic")
    c2.metric("Operating threshold", f"{status['comparison_performance']['operating_threshold']:.2%}")
    c3.metric("Monitoring mode", "Historical demonstration")
    c4.metric("Overall drift status", str(status["overall_drift_status"]).upper())

    period_table = pd.DataFrame(
        [
            {
                "Period": "Development reference",
                "Dates": status["reference_period"],
                "Rows": status["reference_rows"],
                "Role": "Reference distribution",
            },
            {
                "Period": "Final holdout",
                "Dates": status["comparison_period"],
                "Rows": status["comparison_rows"],
                "Role": "Later comparison + labelled evaluation",
            },
        ]
    )
    st.dataframe(period_table, use_container_width=True, hide_index=True)

    with st.expander("Monitoring traceability"):
        st.json(
            {
                "model_name": status["model_name"],
                "reference_rows": status["reference_rows"],
                "comparison_rows": status["comparison_rows"],
                "feature_count": population["feature_count"],
                "comparison_labels_reliable": population["comparison"]["labels_reliable"],
            }
        )

with tabs[1]:
    st.subheader("Data and prediction drift")
    st.write(
        "The project implementation and Evidently AI are shown separately. Differences are "
        "kept visible because PSI implementations can vary in binning and edge handling."
    )

    st.markdown("#### Project Python monitoring")
    p1, p2, p3 = st.columns(3)
    p1.metric("Variables at escalate", int((project_all["severity"] == "escalate").sum()))
    p2.metric("Variables at watch", int((project_all["severity"] == "watch").sum()))
    p3.metric("Prediction-score PSI", f"{status['prediction_drift']['psi']:.3f}")

    st.info(
        "Project thresholds: PSI below 0.10 = OK; 0.10 to below 0.25 = WATCH; "
        "0.25 or above = ESCALATE. These are project-specific monitoring thresholds."
    )

    fig = px.bar(
        project_all,
        x="feature",
        y="psi",
        title="Project PSI by monitored variable",
        labels={"feature": "", "psi": "PSI"},
    )
    fig.add_hline(y=0.10, line_dash="dash", annotation_text="WATCH 0.10")
    fig.add_hline(y=0.25, line_dash="dot", annotation_text="ESCALATE 0.25")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        project_all.rename(
            columns={
                "feature": "column",
                "psi": "project_psi",
                "ks_statistic": "ks_statistic",
                "severity": "project_severity",
            }
        ).sort_values("project_psi", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Evidently AI monitoring")
    st.caption(
        "Independent PSI-based cross-check on the same monitored variables. "
        "The retained Evidently evidence comes from the validated companion analysis."
    )
    if evidently.empty:
        st.warning("No Evidently ValueDrift metrics were found.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Variables Evidently flags as drifted", int(evidently["evidently_drift"].sum()))
        e2.metric("Variables not flagged", int((~evidently["evidently_drift"]).sum()))
        score_match = evidently.loc[evidently["column"] == PREDICTION_COLUMN]
        score_value = float(score_match.iloc[0]["evidently_psi"]) if not score_match.empty else float("nan")
        e3.metric("Prediction-score PSI", f"{score_value:.3f}")

        fig_e = px.bar(
            evidently,
            x="column",
            y="evidently_psi",
            title="Evidently AI PSI by monitored variable",
            labels={"column": "", "evidently_psi": "PSI"},
        )
        fig_e.add_hline(y=DRIFT_THRESHOLD, line_dash="dash", annotation_text="DRIFT 0.10")
        st.plotly_chart(fig_e, use_container_width=True)
        st.dataframe(
            evidently.sort_values("evidently_psi", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        project_table = project_all.rename(columns={"feature": "column", "psi": "project_psi"})
        comparison = project_table.merge(
            evidently[["column", "evidently_psi", "evidently_drift"]],
            on="column",
            how="inner",
        )
        comparison["project_drift_at_0_10"] = comparison["project_psi"] >= DRIFT_THRESHOLD
        agreed = comparison.loc[
            comparison["project_drift_at_0_10"] & comparison["evidently_drift"], "column"
        ].tolist()
        project_only = comparison.loc[
            comparison["project_drift_at_0_10"] & ~comparison["evidently_drift"], "column"
        ].tolist()
        evidently_only = comparison.loc[
            ~comparison["project_drift_at_0_10"] & comparison["evidently_drift"], "column"
        ].tolist()

        st.markdown("#### Interpretation")
        if agreed:
            st.success("Agreement on drift: " + ", ".join(agreed) + ".")
        if project_only:
            st.info("Project-only signals: " + ", ".join(project_only) + ".")
        if evidently_only:
            st.info("Evidently-only signals: " + ", ".join(evidently_only) + ".")

    st.markdown("#### Prediction-score comparison")
    p1, p2 = st.columns(2)
    p1.metric("Reference mean risk", f"{status['prediction_drift']['reference_mean']:.2%}")
    p2.metric("Final-holdout mean risk", f"{status['prediction_drift']['current_mean']:.2%}")

with tabs[2]:
    st.subheader("Performance and calibration")
    perf = status["comparison_performance"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{perf['roc_auc']:.3f}")
    c2.metric("Recall", f"{perf['recall']:.1%}")
    c3.metric("Precision", f"{perf['precision']:.1%}")
    c4.metric("Top-20% capture", f"{perf['top20_capture']:.1%}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Brier score", f"{perf['brier']:.3f}")
    c6.metric("ECE (10 bins)", f"{perf['ece_10bin']:.3f}")
    c7.metric("F1", f"{perf['f1']:.3f}")

    st.caption(
        "These metrics use the chronologically later final holdout and the frozen operating threshold."
    )
    st.info(status["interpretation"])

with tabs[3]:
    st.subheader("Fairness feasibility and operational robustness")
    st.error(
        "**Demographic fairness cannot be demonstrated from this dataset.** Reliable protected-group "
        "attributes are unavailable, so the project does not infer or manufacture demographic groups."
    )

    st.markdown("#### What the monitoring evidence can support")
    st.write(
        "- Drift in observed model features and prediction scores.\n"
        "- Performance and calibration on the trusted final holdout.\n"
        "- Representation and performance checks across defensible operational segments.\n"
        "- Ongoing visibility of proxy, representation and operational-use risks."
    )

    st.markdown("#### What it cannot support")
    st.write(
        "- Demographic parity, Equalized Odds or disparate-impact claims for protected groups.\n"
        "- A conclusion that the model is fair simply because protected attributes are absent.\n"
        "- Future production performance before trustworthy outcome labels have matured."
    )

    st.markdown("#### Monitoring governance")
    st.write(
        "Operational segments are robustness and representation checks, not demographic fairness groups. "
        "Drift is an investigation signal and should not be treated as proof of model failure."
    )
    st.caption(status["fairness_boundary"])
