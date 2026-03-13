#!/usr/bin/env python3
"""Build a family-level generalization compass with leave-one-family-out retraining."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.diagnostics import DEFAULT_REFERENCE_OOF_SPECS, run_family_generalization_compass
from churn_baseline.feature_engineering import normalize_feature_blocks


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze family generalization with leave-one-family-out retraining")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--family-level",
        choices=("segment3", "segment5"),
        default="segment5",
        help="Family granularity used for selection and holdout.",
    )
    parser.add_argument(
        "--feature-blocks",
        default="H,R,V",
        help="Feature blocks used for leave-one-family-out retraining.",
    )
    parser.add_argument(
        "--reference-oof",
        default="",
        help="Optional direct reference OOF spec: <path>[#<prediction_column>]. Overrides --oof + --reference-weights-json.",
    )
    parser.add_argument(
        "--oof",
        action="append",
        default=[],
        help="OOF input spec: <name>=<path>[#<prediction_column>]. Defaults to cb/xgb/lgb/r/rv teacher blend.",
    )
    parser.add_argument(
        "--reference-weights-json",
        default="artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json",
        help="Weights JSON for the default weighted teacher reference.",
    )
    parser.add_argument("--top-k-families", type=int, default=12, help="Maximum number of families to evaluate with LOFO.")
    parser.add_argument("--min-train-rows", type=int, default=5000, help="Minimum train rows for a family to be eligible.")
    parser.add_argument("--min-test-rows", type=int, default=2000, help="Minimum test rows for a family to be eligible.")
    parser.add_argument("--iterations", type=int, default=400, help="Fixed CatBoost iterations for LOFO retraining.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for LOFO retraining.")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth for LOFO retraining.")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="L2 regularization for LOFO retraining.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for LOFO retraining.")
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/diagnostic_family_generalization_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--out-csv",
        default="artifacts/reports/diagnostic_family_generalization_families.csv",
        help="Output CSV table path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_blocks = _parse_feature_blocks(args.feature_blocks)
    oof_specs = args.oof if args.oof else list(DEFAULT_REFERENCE_OOF_SPECS)
    summary = run_family_generalization_compass(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        family_level=args.family_level,
        feature_blocks=feature_blocks,
        top_k_families=args.top_k_families,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
        reference_oof_spec=args.reference_oof.strip() or None,
        oof_specs=oof_specs,
        reference_weights_json=args.reference_weights_json,
        out_json_path=args.out_json,
        out_csv_path=args.out_csv,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
