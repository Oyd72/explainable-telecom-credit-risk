"""Streamlit dashboard for drift and performance evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "reports/monitoring/latest_status.json"
DRIFT = ROOT / "reports/monitoring/feature_drift.csv"

st.set_page_config(page_title="Credit-risk monitoring", page_icon="📈", layout="wide")
st.title("Credit-risk monitoring")
st.caption("Distribution drift, performance and evidence limits")

if not STATUS.exists() or not DRIFT.exists():
    st.warning(
        "Run `python -m src.monitoring.prepare` and `python -m src.monitoring.run`."
    )
    st.stop()

status = json.loads(STATUS.read_text(encoding="utf-8"))
drift = pd.read_csv(DRIFT)

c1, c2, c3 = st.columns(3)
c1.metric("Overall drift status", str(status["overall_drift_status"]).upper())
c2.metric("Reference rows", f"{status['reference_rows']:,}")
c3.metric("Comparison rows", f"{status['comparison_rows']:,}")

prediction = status["prediction_drift"]
c1, c2, c3 = st.columns(3)
c1.metric("Prediction PSI", f"{prediction['psi']:.3f}")
c2.metric("Prediction KS", f"{prediction['ks_statistic']:.3f}")
c3.metric("Mean comparison risk", f"{prediction['current_mean']:.2%}")

st.subheader("Feature drift")
fig = px.bar(
    drift.sort_values("psi"),
    x="psi",
    y="feature",
    orientation="h",
    labels={"psi": "Population Stability Index", "feature": ""},
)
fig.add_vline(x=0.10, line_dash="dash")
fig.add_vline(x=0.25, line_dash="dot")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    drift.sort_values("psi", ascending=False),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Later-period model performance")
metrics = status["comparison_performance"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
c2.metric("Average precision", f"{metrics['average_precision']:.3f}")
c3.metric("Top-20% capture", f"{metrics['top20_capture']:.1%}")
c4.metric("Brier score", f"{metrics['brier']:.3f}")

st.info("Drift is an investigation signal, not proof of model failure.")
st.warning(
    "Protected demographic attributes are unavailable; "
    "this dashboard does not make demographic fairness claims."
)
