"""Prepare reference and comparison populations for monitoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/telecom_delinquency_model_ready.csv"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/model.yaml"),
    )
    parser.add_argument(
        "--monitoring-config",
        type=Path,
        default=Path("config/monitoring.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/monitoring"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/monitoring/population_summary.json"),
    )
    args = parser.parse_args()

    model_config = yaml.safe_load((PROJECT_ROOT / args.model_config).read_text(encoding="utf-8"))
    monitoring_config = yaml.safe_load((PROJECT_ROOT / args.monitoring_config).read_text(encoding="utf-8"))

    df = pd.read_csv(PROJECT_ROOT / args.data)
    df["pdate"] = pd.to_datetime(df["pdate"], errors="raise")

    features = list(model_config["features"])
    required = {"pdate", "delinquent_5d", *features}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Model-ready data are missing required columns: {missing}")

    reference_end = pd.Timestamp(monitoring_config["reference"]["end"])
    comparison_start = pd.Timestamp(monitoring_config["comparison"]["start"])
    comparison_end = pd.Timestamp(monitoring_config["comparison"]["end"])

    reference = df.loc[df["pdate"] <= reference_end].copy()
    comparison = df.loc[
        (df["pdate"] >= comparison_start) & (df["pdate"] <= comparison_end)
    ].copy()

    columns = ["pdate", *features, "delinquent_5d"]
    reference = reference[columns]
    comparison = comparison[columns]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output_dir / "reference.csv", index=False)
    comparison.to_csv(output_dir / "comparison_labelled.csv", index=False)

    summary = {
        "source_rows": int(len(df)),
        "feature_count": len(features),
        "features": features,
        "reference": {
            "start": df["pdate"].min().date().isoformat(),
            "end": reference_end.date().isoformat(),
            "rows": int(len(reference)),
        },
        "comparison": {
            "start": comparison_start.date().isoformat(),
            "end": comparison_end.date().isoformat(),
            "rows": int(len(comparison)),
            "labels_reliable": True,
        },
        "note": (
            "This historical comparison demonstrates the monitoring design. "
            "Future scored batches can use the same label-free checks immediately, "
            "with outcome-based metrics added after trustworthy labels mature."
        ),
    }

    summary_path = PROJECT_ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
