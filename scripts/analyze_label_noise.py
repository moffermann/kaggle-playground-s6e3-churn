#!/usr/bin/env python3
"""Audit label noise, near-duplicates, and hard examples against incumbent v3."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.diagnostics import DEFAULT_REFERENCE_OOF_SPECS
from churn_baseline.noise_audit import DEFAULT_V3_OOF_SPEC, run_label_noise_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit label noise and near-duplicates against incumbent v3")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--v3-oof",
        default=DEFAULT_V3_OOF_SPEC,
        help="Direct v3 OOF spec: <path>[#<prediction_column>].",
    )
    parser.add_argument(
        "--oof",
        action="append",
        default=[],
        help="Optional teacher OOF component spec: <name>=<path>[#<prediction_column>]. Defaults to cb/xgb/lgb/r/rv.",
    )
    parser.add_argument("--high-confidence-upper", type=float, default=0.90, help="Upper threshold for confident wrong negatives.")
    parser.add_argument("--high-confidence-lower", type=float, default=0.10, help="Lower threshold for confident wrong positives.")
    parser.add_argument("--hard-loss-quantile", type=float, default=0.995, help="Quantile used to mark hard examples by logloss.")
    parser.add_argument("--low-disagreement-quantile", type=float, default=0.35, help="Teacher std quantile used to mark low disagreement.")
    parser.add_argument("--high-disagreement-quantile", type=float, default=0.75, help="Teacher std quantile used as instability proxy.")
    parser.add_argument(
        "--max-near-duplicate-group-size",
        type=int,
        default=5,
        help="Maximum coarse duplicate group size still considered a near-duplicate conflict.",
    )
    parser.add_argument("--top-n-families", type=int, default=15, help="How many top segment3/segment5 groups to report.")
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/label_noise_audit_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--out-rows-csv",
        default="artifacts/reports/label_noise_audit_suspicious_rows.csv",
        help="Output CSV path for suspicious rows.",
    )
    parser.add_argument(
        "--out-duplicate-csv",
        default="artifacts/reports/label_noise_audit_duplicate_groups.csv",
        help="Output CSV path for duplicate groups.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    teacher_specs = args.oof if args.oof else list(DEFAULT_REFERENCE_OOF_SPECS)
    summary = run_label_noise_audit(
        train_csv_path=args.train_csv,
        v3_oof_spec=args.v3_oof,
        teacher_oof_specs=teacher_specs,
        high_confidence_upper=args.high_confidence_upper,
        high_confidence_lower=args.high_confidence_lower,
        hard_loss_quantile=args.hard_loss_quantile,
        low_disagreement_quantile=args.low_disagreement_quantile,
        high_disagreement_quantile=args.high_disagreement_quantile,
        max_near_duplicate_group_size=args.max_near_duplicate_group_size,
        top_n_families=args.top_n_families,
        out_json_path=args.out_json,
        out_rows_csv_path=args.out_rows_csv,
        out_duplicate_csv_path=args.out_duplicate_csv,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
