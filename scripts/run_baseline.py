#!/usr/bin/env python3
"""Run end-to-end baseline: train, make submission, optional Kaggle submit."""

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import CatBoostHyperParams
from churn_baseline.kaggle_api import submit_file
from churn_baseline.pipeline import make_submission, train_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run churn baseline end-to-end")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--model-path", default="artifacts/models/catboost_baseline.cbm")
    parser.add_argument("--metrics-path", default="artifacts/reports/train_baseline_metrics.json")
    parser.add_argument("--output-csv", default="artifacts/submissions/playground-series-s6e3.csv")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=2200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=120)
    parser.add_argument("--verbose", type=int, default=200)
    parser.add_argument("--submit", action="store_true", help="Submit the generated CSV to Kaggle")
    parser.add_argument("--competition", default="playground-series-s6e3")
    parser.add_argument("--message", default="playground-series-s6e3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = CatBoostHyperParams(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
    )

    training_result = train_baseline(
        train_csv_path=args.train_csv,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        params=params,
        valid_size=args.valid_size,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
    )
    submission_result = make_submission(
        model_path=args.model_path,
        test_csv_path=args.test_csv,
        output_csv_path=args.output_csv,
    )

    result = {
        "training": training_result,
        "submission": submission_result,
    }

    if args.submit:
        if not Path(args.output_csv).is_file():
            raise FileNotFoundError(f"Submission file not found: {args.output_csv}")
        result["kaggle_submit"] = submit_file(
            competition=args.competition,
            file_path=args.output_csv,
            message=args.message,
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
