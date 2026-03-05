#!/usr/bin/env python3
"""Generate submission CSV from a trained model."""

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.pipeline import make_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission CSV for churn competition")
    parser.add_argument("--model-path", default="artifacts/models/catboost_baseline.cbm")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument(
        "--output-csv",
        default="artifacts/submissions/playground-series-s6e3.csv",
        help="Output submission file path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = make_submission(
        model_path=args.model_path,
        test_csv_path=args.test_csv,
        output_csv_path=args.output_csv,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
