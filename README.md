# Explainable telecom credit risk

Explainable and monitorable short-horizon repayment risk modelling using public telecom microcredit data.

## Project purpose

This project explores how short-horizon repayment risk can be estimated from behavioural transaction data in a way that remains explainable, reproducible and monitorable over time. The model is designed to support prioritisation of higher-risk cases for follow-up, not to make automated lending decisions. The project also examines how temporal drift, calibration and governance affect whether a model that performs well statistically remains suitable for operational use.

## Data

The project uses a publicly available telecom microcredit dataset from Kaggle. Raw customer-level data are not stored in this repository. Reproduction instructions will explain how to obtain the source dataset and place it in the expected local directory before running the pipeline.

The unexplained change in outcome labels after 23 July 2016 is treated as a data-quality and temporal-validity issue, not as ordinary supervised ground truth.

## Repository structure

The repository is organised around data preparation, modelling, explainability, monitoring, inference, deployment and governance. The initial structure is being created before selected components are rebuilt from the earlier prototype.


## Modelling

Model development uses expanding chronological folds instead of random train/test mixing. Logistic regression provides an interpretable baseline, with Random Forest and XGBoost as nonlinear candidates.

The current reference specification is a 12-feature Random Forest with isotonic calibration. It is trained through 6 July 2016, calibrated on 7–13 July, and evaluated on a final holdout covering 14–23 July.

Run the benchmark with:

```bash
python -m src.modeling.benchmark
```

Fit and evaluate the reference model with:

```bash
python -m src.modeling.train
```

The model card is available at `docs/model_card.md`.


## Explainability

The reference model is explained with global and local SHAP analysis plus constrained counterfactual-style examples. SHAP describes how the Random Forest uses its inputs; it is not interpreted causally. Counterfactuals are restricted to observed training-range values and are used only to inspect model sensitivity around the operating threshold.

Run:

```bash
python -m src.explainability.shap_analysis
python -m src.explainability.counterfactual
```

Method details are documented in `docs/explainability.md`.


## Monitoring

The project separates label-free drift detection from outcome-based performance monitoring. Feature and prediction distributions can be checked immediately; ROC-AUC, calibration and error metrics are calculated only when trustworthy labels are available.

Run:

```bash
python -m src.monitoring.prepare
python -m src.monitoring.run
```

An optional Evidently cross-check is available through `src/monitoring/evidently_check.py`. The monitoring design is documented in `docs/monitoring.md`.


## Governance and lifecycle

The project includes a lightweight governance layer covering intended use, fairness limits, impact assessment, retraining, incident response and model retirement. Passing a technical metric does not automatically approve a new model or a new use case.

Key documents are under `docs/governance/`. Governed retraining is implemented in `src/modeling/retrain.py` with automatic promotion disabled.
