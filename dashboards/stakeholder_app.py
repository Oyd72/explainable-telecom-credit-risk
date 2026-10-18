"""Stakeholder-facing Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports/tables/model_evaluation.json"
SHAP = ROOT / "reports/tables/shap_global_importance.csv"
COUNTERFACTUAL = ROOT / "reports/tables/counterfactual_summary.json"

st.set_page_config(
    page_title="Explainable telecom credit risk",
    page_icon="📊",
    layout="wide",
)

st.title("Explainable telecom credit risk")
st.caption("Short-horizon repayment-risk modelling with calibration, explanations and monitoring")
st.info(
    "The model supports prioritisation of follow-up cases. "
    "It is not an automated credit approval or refusal system."
)

tabs = st.tabs(["Overview", "Explainability", "Counterfactuals", "Limits"])

with tabs[0]:
    st.subheader("Reference model")
    if EVAL.exists():
        payload = json.loads(EVAL.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        c2.metric("Average precision", f"{metrics['average_precision']:.3f}")
        c3.metric("Top-20% capture", f"{metrics['top20_capture']:.1%}")
        c4.metric("Brier score", f"{metrics['brier']:.3f}")
        st.write(
            f"The evaluation uses {payload['holdout_rows']:,} chronologically later records. "
            f"The operating threshold is {payload['threshold']:.4f}."
        )
    else:
        st.warning("Run `python -m src.modeling.train` to generate evaluation evidence.")

with tabs[1]:
    st.subheader("Global SHAP importance")
    if SHAP.exists():
        importance = pd.read_csv(SHAP)
        fig = px.bar(
            importance.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            labels={"mean_abs_shap": "Mean absolute SHAP value", "feature": ""},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("SHAP describes fitted model behaviour; it does not establish causation.")
    else:
        st.warning("Run `python -m src.explainability.shap_analysis`.")

with tabs[2]:
    st.subheader("Counterfactual-style examples")
    if COUNTERFACTUAL.exists():
        st.json(json.loads(COUNTERFACTUAL.read_text(encoding="utf-8")))
        st.caption("These examples are model-behaviour explanations, not customer advice.")
    else:
        st.warning("Run `python -m src.explainability.counterfactual`.")

with tabs[3]:
    st.subheader("Important limits")
    st.markdown(
        "- Short historical period\n"
        "- Material temporal movement in several features\n"
        "- Unexplained all-success label regime after 23 July 2016\n"
        "- No reliable protected demographic attributes\n"
        "- Not intended for autonomous lending decisions"
    )
