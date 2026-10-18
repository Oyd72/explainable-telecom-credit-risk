# Model artefacts

Model binaries are generated locally and are not committed by default.

Run:

```bash
python -m src.modeling.train
```

This writes:

- `models/repayment_risk_model.joblib`
- `models/model_metadata.json`
- `reports/tables/model_evaluation.json`

The packaged object contains the fitted median imputer, Random Forest, isotonic calibrator, feature order and operating threshold. The metadata file records a SHA-256 fingerprint for the generated binary.

The final holdout is used only for evaluation, not fitting or calibration.
