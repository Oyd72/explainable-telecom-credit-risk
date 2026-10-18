# Explainability approach

The project uses two complementary explanation methods.

## SHAP

SHAP is used to inspect the fitted Random Forest globally and locally. The global analysis ranks features by mean absolute SHAP value and also produces a beeswarm view of direction and spread. Local waterfall plots show how feature values contribute to representative low-, typical- and high-risk predictions.

SHAP is applied to the Random Forest before isotonic calibration. This is intentional: calibration changes the probability scale but does not change the underlying tree splits. Local outputs report both raw and calibrated risk so that distinction remains visible.

SHAP values describe the model. They do not establish that changing a feature would cause repayment behaviour to change.

## Counterfactual-style analysis

The counterfactual script looks for small changes, constrained to values observed in the training distribution, that would move a selected case across the operating threshold.

It searches single-feature changes first. If none are found, it searches two-feature combinations among the strongest global SHAP features.

These results are contrastive explanations of model behaviour. They are not recommendations to customers and should not be interpreted causally.

## Outputs

Running:

```bash
python -m src.explainability.shap_analysis
python -m src.explainability.counterfactual
```

creates explanation tables under `reports/tables/` and figures under `reports/figures/explainability/`.

The counterfactual analysis expects the SHAP global-importance table if available, but can fall back to the configured feature order.
