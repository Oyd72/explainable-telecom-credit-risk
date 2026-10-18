# Data pipeline methodology

The data pipeline is designed around chronology, reproducibility and conservative treatment of uncertain fields.

## Flow

`raw validation → cleaning → interim validation → model-ready transformation → processed validation → representation diagnostics`

Raw validation is diagnostic because the source contains known quality issues that the cleaning stage is expected to address. Interim and processed validation are blocking controls. The final representation step is diagnostic and does not claim demographic fairness because the dataset does not contain reliable protected-group attributes.

## Point-in-time controls

Customer-history features are derived only from transactions on strictly earlier dates. Same-day transactions are treated as contemporaneous. The identifier-like `msisdn` field is retained only while chronology requires it and is removed before model-ready output is written.

Records after 23 July 2016 are excluded from ordinary supervised modelling because every later record is labelled as successful repayment and the source documentation does not explain the abrupt change.

## Reproducibility

The raw Kaggle CSV remains outside version control. Place it at:

`data/raw/sample_data_intw.csv`

The pipeline writes derived data locally under `data/interim/` and `data/processed/`, with aggregate validation and audit evidence under `reports/`.
