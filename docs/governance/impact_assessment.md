# Algorithmic impact assessment

## Purpose and scope

The model estimates the probability that a telecom-enabled microcredit transaction will remain unpaid after five days. Its intended role is risk prioritisation and follow-up support.

The current project does not approve use of the score as the sole basis for automatic credit approval, refusal, pricing or another consequential customer decision.

## Potential benefits

The model may help repayment teams focus limited follow-up capacity on cases with higher estimated short-horizon risk. It also provides a reproducible framework for calibration, explanation and monitoring.

Those benefits depend on proportionate downstream use.

## Principal risks

### Temporal instability

The historical observation window is short, and several inputs move materially across time.

Controls include temporal validation, calibration, drift monitoring and governed retraining.

Residual risk remains because historical performance cannot guarantee future stability.

### Over-reliance

Users may treat a probability as a definitive judgment.

Controls include explicit intended-use statements, human review, dashboard limitations and prohibition of autonomous adverse use.

### Fairness uncertainty

Protected demographic attributes are unavailable.

The project does not claim demographic fairness. Proxy effects remain possible and require separate assessment in any real deployment.

### Privacy and data handling

The source contains behavioural records and an identifier-like field used temporarily for chronology.

The identifier is removed before model-ready output, source data remain outside the repository, and pipeline audit records avoid storing direct identifiers.

### Explainability misuse

SHAP and counterfactual-style outputs may be mistaken for causal evidence.

Documentation explicitly limits them to explanations of fitted model behaviour.

## Contestability

If a model materially influenced a consequential decision, a real deployment should support human reconsideration, correction of inaccurate input data, identification of the model version used, and an appropriate complaint or escalation route.

The repository does not implement a customer-redress process.

## Conclusion

The project supports controlled experimentation and decision-support use. A real deployment would still require fresh validation, organisation-specific legal assessment, defined human-review procedures, appropriate fairness evidence, operational ownership and a functioning incident-response process.
