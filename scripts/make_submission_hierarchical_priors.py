#!/usr/bin/env python3
"""Generate submission CSV for hierarchical target prior experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import ID_COLUMN, TARGET_COLUMN
from churn_baseline.data import load_csv, prepare_test_features
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.modeling import load_model, predict_proba
from churn_baseline.target_priors import load_target_prior_encoder


def _combine_features(
    base_features: pd.DataFrame,
    prior_features: pd.DataFrame,
    *,
    feature_mode: str,
) -> pd.DataFrame:
    if feature_mode == "priors_only":
        return prior_features.copy()
    if feature_mode == "raw_plus_priors":
        return pd.concat(
            [base_features.reset_index(drop=True), prior_features.reset_index(drop=True)],
            axis=1,
        )
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate submission CSV for hierarchical target prior experiment"
    )
    parser.add_argument("--test-csv", default="data/raw/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/hierarchical_priors_cv_metrics.json",
        help="Metrics JSON produced by experiment_hierarchical_priors.py",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/submissions/playground-series-s6e3-hier-priors.csv",
        help="Output submission CSV path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = json.loads(Path(args.metrics_path).read_text(encoding="utf-8"))
    feature_blocks = list(normalize_feature_blocks(metrics.get("base_feature_blocks", [])))
    feature_mode = str(metrics.get("feature_mode", "raw_plus_priors"))
    model_path = metrics["model_path"]
    prior_encoder_path = metrics["prior_encoder_path"]

    test_df = load_csv(args.test_csv)
    ids = test_df[ID_COLUMN].copy()
    x_base = prepare_test_features(test_df, drop_id=True, feature_blocks=feature_blocks)
    encoder = load_target_prior_encoder(prior_encoder_path)
    x_prior = encoder.transform(x_base.reset_index(drop=True))
    x_model = _combine_features(x_base.reset_index(drop=True), x_prior, feature_mode=feature_mode)

    model = load_model(model_path)
    predictions = predict_proba(model, x_model)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: predictions.values})
    submission.to_csv(out_path, index=False)

    result = {
        "output_csv": str(out_path),
        "rows": int(len(submission)),
        "feature_mode": feature_mode,
        "base_feature_blocks": feature_blocks,
        "prediction_min": float(submission[TARGET_COLUMN].min()),
        "prediction_max": float(submission[TARGET_COLUMN].max()),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
