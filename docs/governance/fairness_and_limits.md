# Fairness and evidence limits

## What the data support

The dataset does not contain reliable protected demographic attributes. It can support analysis of operational groups such as first-time versus returning borrowers, but those groups are not substitutes for protected characteristics.

The project does not infer protected attributes simply to manufacture fairness metrics.

## Current analysis

Operational representation checks examine:

- group size and share;
- observed delinquency rate;
- composition over time;
- robustness across selected behavioural segments.

The first-time versus returning-borrower split is left-censored because customer history before the start of the dataset is unavailable. Some apparent first-time customers may have earlier transactions outside the extract.

These analyses can reveal instability or concentration of error. They do not establish demographic fairness.

## If protected-group data were available

A real deployment could consider group-specific error rates, calibration, false-positive and false-negative rates, Equal Opportunity, Equalized Odds or disparate-impact measures only after confirming that:

- collection and use of the attributes are lawful and appropriate;
- group definitions are meaningful in the decision context;
- sample sizes are sufficient;
- privacy risk is controlled;
- the selected fairness metric reflects the harm being assessed.

No single fairness metric should be treated as a universal pass/fail test.

## Escalation

A fairness-related alert should trigger investigation. Possible responses include further validation, feature restriction, threshold review, changes to downstream human review, temporary suspension or retirement.

A disparity should not be automatically labelled discrimination without examining data quality, sample size, operational context, threshold effects and plausible proxy mechanisms.
