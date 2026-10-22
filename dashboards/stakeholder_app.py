"""Stakeholder-facing Streamlit dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib import error, request

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports/tables/model_evaluation.json"
SHAP = ROOT / "reports/tables/shap_global_importance.csv"
COUNTERFACTUAL = ROOT / "reports/tables/counterfactual_summary.json"
THRESHOLDS = ROOT / "reports/tables/threshold_tradeoff.csv"
CASES = ROOT / "reports/tables/representative_cases.csv"
ROBUSTNESS = ROOT / "reports/tables/operational_robustness_segments.csv"
FAIRNESS = ROOT / "reports/tables/fairness_robustness_summary.json"
STATUS = ROOT / "reports/monitoring/latest_status.json"

FEATURE_LABELS = {
    "cnt_ma_rech90": "Recharge count — 90 days",
    "daily_decr30": "Daily decrement measure — 30 days",
    "last_rech_date_ma": "Days associated with last main-account recharge",
    "sumamnt_ma_rech90": "Total main-account recharge amount — 90 days",
    "aon": "Account / network tenure",
    "last_rech_amt_ma": "Most recent main-account recharge amount",
    "daily_decr90": "Daily decrement measure — 90 days",
    "sumamnt_ma_rech30": "Total main-account recharge amount — 30 days",
    "medianamnt_ma_rech30": "Median main-account recharge amount — 30 days",
    "medianmarechprebal90": "Median pre-recharge balance — 90 days",
    "rental30": "Rental / activity measure — 30 days",
    "cnt_ma_rech30": "Recharge count — 30 days",
}

FEATURE_HELP = {
    "cnt_ma_rech90": "Number of main-account recharges in the last 90 days.",
    "daily_decr30": "Daily amount spent from the main account, averaged over the last 30 days.",
    "last_rech_date_ma": "Recharge-recency measure from the source data.",
    "sumamnt_ma_rech90": "Total main-account recharge amount over the last 90 days.",
    "aon": "Age on the cellular network in days.",
    "last_rech_amt_ma": "Amount of the most recent main-account recharge.",
    "daily_decr90": "Daily amount spent from the main account, averaged over the last 90 days.",
    "sumamnt_ma_rech30": "Total main-account recharge amount over the last 30 days.",
    "medianamnt_ma_rech30": "Median main-account recharge amount over the last 30 days.",
    "medianmarechprebal90": "Median pre-recharge balance over the last 90 days.",
    "rental30": "30-day balance/activity measure.",
    "cnt_ma_rech30": "Number of main-account recharges in the last 30 days.",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def api_base_url() -> str:
    value = os.getenv("PREDICTION_API_URL", "").strip()
    if value:
        return value.rstrip("/")
    try:
        secret = str(st.secrets.get("PREDICTION_API_URL", "")).strip()
        if secret:
            return secret.rstrip("/")
    except Exception:
        pass
    return "https://explainable-telecom-credit-risk-api.onrender.com"


def api_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        api_base_url() + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with request.urlopen(req, timeout=25) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Prediction API returned HTTP {exc.code}: {detail}") from exc
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "The prediction service is currently unavailable. "
            "The evidence views remain available."
        ) from exc


@st.cache_data
def load_evidence() -> dict:
    return {
        "evaluation": read_json(EVAL),
        "shap": pd.read_csv(SHAP),
        "counterfactual": read_json(COUNTERFACTUAL),
        "thresholds": pd.read_csv(THRESHOLDS),
        "cases": pd.read_csv(CASES),
        "robustness": pd.read_csv(ROBUSTNESS),
        "fairness": read_json(FAIRNESS),
        "status": read_json(STATUS),
    }


st.set_page_config(page_title="Explainable telecom credit risk", page_icon="📊", layout="wide")
st.title("Five-day telecom delinquency")
st.caption("Stakeholder dashboard")
st.warning(
    "**Model status:** demonstration model, not production-ready. "
    "The model supports prioritisation and follow-up; it does not approve or refuse credit."
)

evidence = load_evidence()
evaluation = evidence["evaluation"]
metrics = evaluation["metrics"]
threshold = float(evaluation["threshold"])
robustness = evidence["robustness"]
overall = robustness.loc[robustness["group_type"] == "overall"].iloc[0]

tabs = st.tabs(["Overview", "Prediction explorer", "Explainability & what-if", "Fairness & limits"])

with tabs[0]:
    st.subheader("What the model is for")
    st.write(
        "The model estimates the probability that a telecom-enabled microcredit transaction "
        "will remain unpaid after five days. It is designed to support risk prioritisation "
        "and repayment follow-up, not automated lending decisions."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c2.metric("Top-20% capture", f"{metrics['top20_capture']:.1%}")
    c3.metric("Recall at threshold", f"{metrics['recall']:.1%}")
    c4.metric("Precision at threshold", f"{metrics['precision']:.1%}")
    c5.metric("Observed delinquency", f"{float(overall['delinquency_rate']):.1%}")

    st.caption(
        "Reviewing the 20% of cases with the highest predicted risk captures about "
        f"{metrics['top20_capture']:.1%} of delinquent cases in the final holdout."
    )

    st.markdown("#### Explore the threshold trade-off")
    tradeoff = evidence["thresholds"]
    scenario = st.slider(
        "Illustrative threshold",
        min_value=float(tradeoff["threshold"].min()),
        max_value=float(tradeoff["threshold"].max()),
        value=0.18,
        step=0.01,
        format="%.2f",
    )
    row = tradeoff.iloc[(tradeoff["threshold"] - scenario).abs().argsort()[:1]].iloc[0]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Illustrative cut-off", f"{row['threshold']:.0%}")
    t2.metric("Precision", f"{row['precision']:.1%}")
    t3.metric("Recall", f"{row['recall']:.1%}")
    t4.metric("Cases flagged", f"{row['flagged_rate']:.1%}")

    long = tradeoff.melt(
        id_vars="threshold",
        value_vars=["precision", "recall"],
        var_name="Metric",
        value_name="Value",
    )
    long["Metric"] = long["Metric"].str.title()
    fig = px.line(
        long,
        x="threshold",
        y="Value",
        color="Metric",
        title="Precision and recall across alternative thresholds",
    )
    fig.add_vline(x=threshold, line_dash="dash", annotation_text="Frozen threshold 17.51%")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "The alternatives are descriptive scenarios only. The operating threshold was "
        "fixed before the final holdout was evaluated."
    )

    with st.expander("Model identity and traceability"):
        st.json(
            {
                "model_name": evidence["status"]["model_name"],
                "feature_count": evaluation["feature_count"],
                "training_rows": evaluation["training_rows"],
                "calibration_rows": evaluation["calibration_rows"],
                "holdout_rows": evaluation["holdout_rows"],
                "operating_threshold": threshold,
            }
        )

with tabs[1]:
    st.subheader("Prediction explorer")
    st.write(
        "Enter the information available for a transaction and request a calibrated five-day "
        "delinquency probability from the deployed FastAPI model service."
    )

    case_names = {
        "Blank form": None,
        "Low-risk example": "low_risk",
        "Typical-risk example": "typical_risk",
        "High-risk example": "high_risk",
    }
    selected = st.selectbox("Optional starting point", list(case_names))
    preset = {}
    if case_names[selected]:
        rows = evidence["cases"].loc[evidence["cases"]["case"] == case_names[selected]]
        preset = dict(zip(rows["feature"], rows["feature_value"]))

    values: dict[str, float | None] = {}
    features = list(FEATURE_LABELS)
    left, right = st.columns(2)
    for index, feature in enumerate(features):
        container = left if index % 2 == 0 else right
        default = preset.get(feature)
        raw = container.text_input(
            FEATURE_LABELS[feature],
            value="" if default is None else str(default),
            help=f"{FEATURE_HELP[feature]} Model field: {feature}",
        )
        if not raw.strip():
            values[feature] = None
        else:
            try:
                values[feature] = float(raw)
            except ValueError:
                values[feature] = None
                container.caption("Enter a numeric value or leave the field blank.")

    if st.button("Calculate predicted risk", type="primary"):
        try:
            result = api_json("/predict", values)
            calibrated = float(result["calibrated_delinquency_probability"])
            st.metric("Predicted five-day delinquency risk", f"{calibrated:.1%}")
            st.caption(f"Operating threshold: {threshold:.2%}")

            if bool(result["higher_risk_flag"]):
                st.warning(
                    "**Higher-risk follow-up flag.** This is a prioritisation signal, not a credit decision."
                )
            else:
                st.success(
                    "**No higher-risk follow-up flag.** This does not guarantee on-time repayment."
                )

            info = api_json("/model-info")
            medians = info["imputation_medians"]
            sensitivity = []
            for feature, value in values.items():
                if value is None:
                    continue
                changed = dict(values)
                changed[feature] = float(medians[feature])
                changed_result = api_json("/predict", changed)
                changed_score = float(changed_result["calibrated_delinquency_probability"])
                sensitivity.append(
                    {
                        "feature": feature,
                        "risk_change": calibrated - changed_score,
                        "abs_risk_change": abs(calibrated - changed_score),
                    }
                )
            if sensitivity:
                local = pd.DataFrame(sensitivity).nlargest(5, "abs_risk_change")
                local["label"] = local["feature"].map(FEATURE_LABELS)
                chart = px.bar(
                    local.sort_values("abs_risk_change"),
                    x="abs_risk_change",
                    y="label",
                    orientation="h",
                    title="Local sensitivity compared with the model's training median",
                )
                st.plotly_chart(chart, use_container_width=True)
                st.caption(
                    "One-feature-at-a-time sensitivity is descriptive model behaviour, not a causal explanation."
                )
        except RuntimeError as exc:
            st.error(str(exc))

with tabs[2]:
    st.subheader("What drives the model")
    shap = evidence["shap"].head(8).copy()
    shap["label"] = shap["feature"].map(FEATURE_LABELS)
    fig = px.bar(
        shap.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="label",
        orientation="h",
        title="Strongest global model drivers",
        labels={"mean_abs_shap": "Mean absolute SHAP value", "label": ""},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("SHAP describes fitted model behaviour; it does not establish causation.")

    st.markdown("#### Borderline predictions can be sensitive to small changes")
    cases = [c for c in evidence["counterfactual"]["cases"] if c.get("counterfactual_found")]
    cards = st.columns(max(1, len(cases)))
    for card, case in zip(cards, cases):
        change = case["changes"][0]
        with card:
            st.markdown(f"**{FEATURE_LABELS.get(change['feature'], change['feature'])}**")
            st.write(f"Input: {change['from']:,.2f} → {change['to']:,.2f}")
            st.write(
                f"Predicted risk: {case['original_score']:.2%} → "
                f"{case['counterfactual_score']:.2%}"
            )
    st.info(
        "Cases close to the operating threshold can move across the follow-up cut-off after "
        "a modest change in one input. These examples are contrastive checks, not customer advice."
    )

with tabs[3]:
    st.subheader("Fairness and robustness")
    st.error(
        "**Demographic fairness cannot be assessed from this dataset.** Reliable protected-group "
        "attributes are unavailable, so the project does not infer substitute demographic groups."
    )

    tenure = robustness.loc[robustness["group_type"] == "account_tenure_quartile"].copy()
    tenure["segment"] = ["Shortest tenure", "Lower-middle tenure", "Upper-middle tenure", "Longest tenure"]
    fig = px.bar(
        tenure,
        x="segment",
        y="roc_auc",
        range_y=[0.75, 0.88],
        title="Model discrimination across account-tenure segments",
        labels={"segment": "", "roc_auc": "ROC-AUC"},
    )
    st.plotly_chart(fig, use_container_width=True)

    recharge = robustness.loc[robustness["group_type"] == "recharge90_quartile"].copy()
    recharge = recharge.sort_values("delinquency_rate", ascending=False)
    recharge["segment"] = ["0–2 recharges", "2–5 recharges", "5–10 recharges", "10–132 recharges"]
    fig = px.bar(
        recharge,
        x="segment",
        y="roc_auc",
        range_y=[0.60, 0.75],
        title="Model discrimination within 90-day recharge-frequency segments",
        labels={"segment": "", "roc_auc": "ROC-AUC"},
    )
    st.plotly_chart(fig, use_container_width=True)

    fairness = evidence["fairness"]
    with st.container(border=True):
        st.markdown("#### Fairness feasibility and residual risk")
        st.write(fairness["fairness_scope"]["reason"])
        st.write(fairness["fairness_scope"]["proxy_note"])
        st.caption(fairness["interpretation_caution"])

    with st.expander("Important limitations"):
        st.write("- Short historical observation period.")
        st.write("- Material temporal movement in several model inputs.")
        st.write("- Unexplained all-success label regime after 23 July 2016.")
        st.write("- No reliable protected demographic attributes.")
        st.write("- Not intended for autonomous lending decisions.")
