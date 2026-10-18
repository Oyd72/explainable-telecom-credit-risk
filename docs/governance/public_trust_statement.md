# Public trust statement

This project demonstrates an explainable short-horizon repayment-risk model using public historical telecom microcredit data.

The model estimates the probability that a transaction will remain unpaid after five days. It is designed as a prioritisation signal for follow-up, not as an automatic lending decision.

Direct identifiers are not used as model features. The identifier-like customer field needed for chronology is removed before model-ready data are written.

The model is evaluated on a later time period, calibrated separately, and monitored for distribution change. SHAP and constrained counterfactual-style explanations are provided to make model behaviour easier to inspect.

There are important limits. The historical period is short, the data show material temporal movement, and the dataset does not contain reliable protected demographic attributes. The project cannot demonstrate demographic fairness and does not claim to do so.

The repository is intended to show how predictive performance, explainability, monitoring and governance can be considered together. It should not be interpreted as evidence that this model is ready for unrestricted production lending use.
