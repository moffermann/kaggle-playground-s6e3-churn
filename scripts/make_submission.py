#!/usr/bin/env python3
"""Generate submission CSV from a trained model."""

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.pipeline import make_submission


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission CSV for churn competition")
    parser.add_argument("--model-path", default="artifacts/models/catboost_baseline.cbm")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument(
        "--feature-blocks",
        default="none",
        help="Optional feature blocks (A,B,C,G,H,R,S,V,O,P) or none.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/submissions/playground-series-s6e3.csv",
        help="Output submission file path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_blocks = _parse_feature_blocks(args.feature_blocks)
    result = make_submission(
        model_path=args.model_path,
        test_csv_path=args.test_csv,
        output_csv_path=args.output_csv,
        feature_blocks=feature_blocks,
        train_csv_path=args.train_csv,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
