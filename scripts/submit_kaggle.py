#!/usr/bin/env python3
"""Submit a generated CSV to Kaggle competition."""

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.kaggle_api import submit_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit CSV to Kaggle competition")
    parser.add_argument("--competition", default="playground-series-s6e3")
    parser.add_argument(
        "--file-path",
        default="artifacts/submissions/playground-series-s6e3.csv",
        help="Submission file path",
    )
    parser.add_argument("--message", default="playground-series-s6e3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    response = submit_file(
        competition=args.competition,
        file_path=args.file_path,
        message=args.message,
    )
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
