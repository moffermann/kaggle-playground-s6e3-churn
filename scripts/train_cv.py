#!/usr/bin/env python3
"""Train baseline model with Stratified K-Fold CV and save OOF artifacts."""

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import CatBoostHyperParams
from churn_baseline.pipeline import train_baseline_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost baseline with Stratified K-Fold CV")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--model-path",
        default="artifacts/models/catboost_baseline_cv.cbm",
        help="Output model path",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/train_baseline_cv_metrics.json",
        help="Output CV metrics JSON path",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/train_baseline_cv_oof.csv",
        help="Output OOF predictions CSV path",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of Stratified K folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--iterations", type=int, default=2200, help="Max boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="L2 regularization")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=120,
        help="Early stopping rounds per fold",
    )
    parser.add_argument("--verbose", type=int, default=200, help="CatBoost verbose frequency")
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

    metrics = train_baseline_cv(
        train_csv_path=args.train_csv,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
        params=params,
        folds=args.folds,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
