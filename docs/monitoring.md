# Monitoring design

The monitoring layer separates two questions that are often confused:

1. **Has the input or score distribution changed?**
2. **Has predictive performance changed?**

The first can be evaluated without labels. The second requires trustworthy outcomes.

## Reference and comparison periods

For the reproducibility demonstration, the development population through 13 July 2016 is used as the reference distribution. The final holdout from 14–23 July is the later comparison population.

This is a historical demonstration of the monitoring design, not a claim that the holdout is new production data.

## Label-free monitoring

Each model feature is compared using:

- Population Stability Index (PSI);
- two-sample KS statistic;
- change in missingness;
- reference and comparison means.

Prediction-score drift is evaluated using the same PSI and KS logic.

The project uses configurable thresholds:

- PSI below 0.10: `ok`
- PSI from 0.10 to below 0.25: `watch`
- PSI at or above 0.25: `escalate`

These are project operating thresholds, not universal statistical rules.

## Performance monitoring

When labels are trustworthy, the monitoring run also calculates:

- ROC-AUC;
- average precision;
- Brier score;
- expected calibration error;
- top-20% delinquency capture;
- precision, recall and F1 at the operating threshold.

The code refuses to calculate these metrics when labels are declared unreliable.

## Fairness boundary

The dataset does not contain reliable protected demographic attributes. Drift in operational variables can be monitored, but it is not demographic fairness monitoring.

A real deployment would require a separate decision on whether protected-group monitoring is lawful, appropriate and supported by sufficiently reliable data.

## Evidently cross-check

`src/monitoring/evidently_check.py` provides an optional second implementation of drift detection. It is kept separate from the custom calculations so disagreement remains visible instead of being hidden behind one combined status.

## Running the monitoring workflow

```bash
python -m src.monitoring.prepare
python -m src.monitoring.run
```

Optional Evidently cross-check:

```bash
python -m src.monitoring.evidently_check
```
