# Model card — five-day repayment risk

## Purpose

The model estimates the probability that a telecom-enabled microcredit transaction will remain unpaid after five days. Its intended role is prioritisation of higher-risk cases for follow-up. It is not intended to approve or decline credit automatically.

## Model

The current reference model is a 12-feature Random Forest with median imputation and isotonic probability calibration. The model is fitted on records through 6 July 2016, calibrated on 7–13 July, and evaluated on the later 14–23 July holdout.

The holdout remains outside fitting, calibration and threshold selection.

## Features

The model uses 12 behavioural variables describing recharge activity, account tenure, daily decrement and account balance. Direct identifiers are excluded from the model-ready dataset.

The exact feature order is defined in `config/model.yaml`.

## Evaluation

A reproducibility run against the public source dataset produced results close to the original frozen model:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8288 |
| Average precision | 0.5673 |
| Brier score | 0.1230 |
| ECE, 10 bins | 0.0293 |
| Top-20% delinquency capture | 54.06% |
| Precision at threshold 0.1751 | 42.61% |
| Recall at threshold 0.1751 | 77.47% |
| F1 | 0.5498 |

Small differences from earlier frozen results can occur across library versions because tree construction and numerical behaviour can vary slightly. The qualitative conclusion is unchanged.

## Calibration and threshold

Raw Random Forest probabilities are calibrated with isotonic regression using a later calibration window. The operating threshold is 0.1751, selected before the final holdout was evaluated.

The threshold is an operational choice, not an automatic lending rule.

## Limitations

The available observation period is short. Several inputs move materially over time, and all records after 23 July 2016 carry the same successful-repayment label without an explanation in the source documentation. Those later labels are not treated as ordinary supervised ground truth.

The dataset does not contain reliable protected demographic attributes. Demographic fairness cannot be established from these data, and protected groups are not inferred.

The model describes associations in historical behavioural data; it does not establish causation.

## Appropriate use

Reasonable uses include model-development research, risk prioritisation demonstrations, probability calibration analysis, explainability work and monitoring experiments.

The evidence does not support autonomous credit decisions, long-term default prediction, demographic fairness claims, or use on a materially different population without fresh validation.
