#!/usr/bin/env python3
"""Analyze whether the incumbent residual hierarchy v3 is compressible."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.residual_ablation import DEFAULT_ABLATION_OUT_DIR, run_residual_hierarchy_ablation
from churn_baseline.validation_protocol import DOMINANT_MACROFAMILY, SUPPORTED_STAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ablations and compressed versions of the v3 residual hierarchy")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Competition train CSV used by the validation protocol.")
    parser.add_argument("--test-csv", default="data/raw/test.csv", help="Competition test CSV used by the validation protocol.")
    parser.add_argument("--stage", choices=SUPPORTED_STAGES, default="midcap", help="Validation protocol stage applied to each compressed chain.")
    parser.add_argument(
        "--target-family-level",
        choices=["segment3", "segment5"],
        default="segment3",
        help="Family level used for the target-family guardrail.",
    )
    parser.add_argument(
        "--target-family-value",
        default=DOMINANT_MACROFAMILY,
        help="Target family key evaluated in the protocol guardrails.",
    )
    parser.add_argument(
        "--dominant-family-value",
        default=DOMINANT_MACROFAMILY,
        help="Dominant macrofamily key used for Split B checks.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_ABLATION_OUT_DIR, help="Directory where summary, CSV, and per-chain artifacts are written.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_residual_hierarchy_ablation(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        stage=args.stage,
        target_family_level=args.target_family_level,
        target_family_value=args.target_family_value,
        dominant_family_value=args.dominant_family_value,
        out_dir=args.out_dir,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
