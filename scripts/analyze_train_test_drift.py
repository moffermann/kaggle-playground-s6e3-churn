#!/usr/bin/env python3
"""Analyze train/test drift and adversarial validation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.data import load_csv, prepare_test_features, prepare_train_features
from churn_baseline.diagnostics import (
    build_categorical_drift_table,
    build_numeric_drift_table,
    parse_feature_blocks_arg,
    run_adversarial_validation,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train/test drift diagnostics")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--feature-blocks",
        default="none",
        help="Optional feature blocks (A,B,C,H,R,S,V,O,P) or none.",
    )
    parser.add_argument(
        "--psi-bins",
        type=int,
        default=10,
        help="Quantile bins used for numeric PSI.",
    )
    parser.add_argument("--adv-folds", type=int, default=3, help="Adversarial CV folds.")
    parser.add_argument(
        "--adv-sample-frac",
        type=float,
        default=0.35,
        help="Sample fraction per split for adversarial training.",
    )
    parser.add_argument(
        "--adv-iterations",
        type=int,
        default=500,
        help="CatBoost max iterations for adversarial validation.",
    )
    parser.add_argument(
        "--adv-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for adversarial validation.",
    )
    parser.add_argument("--adv-depth", type=int, default=6, help="Tree depth for adversarial model.")
    parser.add_argument(
        "--adv-early-stopping-rounds",
        type=int,
        default=80,
        help="Early stopping rounds for adversarial folds.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many top drift columns/features to include in summary.",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/diagnostic_train_test_drift.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--out-numeric-csv",
        default="artifacts/reports/diagnostic_train_test_numeric_drift.csv",
        help="Output numeric drift CSV path.",
    )
    parser.add_argument(
        "--out-categorical-csv",
        default="artifacts/reports/diagnostic_train_test_categorical_drift.csv",
        help="Output categorical drift CSV path.",
    )
    return parser.parse_args()


def _adv_severity(auc: float) -> str:
    if auc >= 0.75:
        return "severe"
    if auc >= 0.65:
        return "high"
    if auc >= 0.58:
        return "moderate"
    return "low"


def main() -> int:
    args = parse_args()
    feature_blocks = parse_feature_blocks_arg(args.feature_blocks)

    train_df = load_csv(args.train_csv)
    test_df = load_csv(args.test_csv)
    x_train, _ = prepare_train_features(train_df, drop_id=True, feature_blocks=feature_blocks)
    x_test = prepare_test_features(test_df, drop_id=True, feature_blocks=feature_blocks)

    numeric_drift = build_numeric_drift_table(x_train, x_test, psi_bins=args.psi_bins)
    categorical_drift = build_categorical_drift_table(x_train, x_test)
    adversarial = run_adversarial_validation(
        x_train,
        x_test,
        folds=args.adv_folds,
        random_state=args.random_state,
        iterations=args.adv_iterations,
        learning_rate=args.adv_learning_rate,
        depth=args.adv_depth,
        early_stopping_rounds=args.adv_early_stopping_rounds,
        sample_frac=args.adv_sample_frac,
    )

    numeric_out = Path(args.out_numeric_csv)
    numeric_out.parent.mkdir(parents=True, exist_ok=True)
    numeric_drift.to_csv(numeric_out, index=False)

    categorical_out = Path(args.out_categorical_csv)
    categorical_out.parent.mkdir(parents=True, exist_ok=True)
    categorical_drift.to_csv(categorical_out, index=False)

    top_k = max(int(args.top_k), 1)
    top_numeric = numeric_drift.head(top_k).to_dict(orient="records")
    top_categorical = categorical_drift.head(top_k).to_dict(orient="records")

    high_psi_count = int((numeric_drift["psi"].fillna(0.0) >= 0.2).sum()) if not numeric_drift.empty else 0
    high_tvd_count = (
        int((categorical_drift["tvd"].fillna(0.0) >= 0.2).sum()) if not categorical_drift.empty else 0
    )
    adv_auc = float(adversarial["cv_mean_auc"])

    summary = {
        "generated_at_utc": utc_now_iso(),
        "inputs": {
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "feature_blocks": feature_blocks,
            "psi_bins": int(args.psi_bins),
            "adv_folds": int(args.adv_folds),
            "adv_sample_frac": float(args.adv_sample_frac),
            "adv_iterations": int(args.adv_iterations),
            "adv_learning_rate": float(args.adv_learning_rate),
            "adv_depth": int(args.adv_depth),
            "adv_early_stopping_rounds": int(args.adv_early_stopping_rounds),
            "random_state": int(args.random_state),
        },
        "dataset": {
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "feature_count": int(x_train.shape[1]),
        },
        "adversarial_validation": {
            **adversarial,
            "severity": _adv_severity(adv_auc),
        },
        "numeric_drift": {
            "row_count": int(numeric_drift.shape[0]),
            "high_psi_count_ge_0_2": high_psi_count,
            "top_rows": top_numeric,
            "csv_path": str(numeric_out),
        },
        "categorical_drift": {
            "row_count": int(categorical_drift.shape[0]),
            "high_tvd_count_ge_0_2": high_tvd_count,
            "top_rows": top_categorical,
            "csv_path": str(categorical_out),
        },
    }
    write_json(args.out_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
