#!/usr/bin/env python3
"""Train CatBoost baseline with CV across multiple seeds."""

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import CatBoostHyperParams
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.pipeline import train_baseline_cv_multiseed


def _parse_seeds(raw: str) -> list[int]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("At least one seed is required")
    return [int(token) for token in tokens]


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost CV multi-seed ensemble")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--models-dir",
        default="artifacts/models/multiseed",
        help="Output directory for per-seed models",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/train_cv_multiseed_metrics.json",
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/train_cv_multiseed_oof.csv",
        help="Output OOF predictions CSV path",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of Stratified K folds")
    parser.add_argument(
        "--seeds",
        default="42,2024,3407",
        help="Comma-separated list of integer seeds",
    )
    parser.add_argument(
        "--feature-blocks",
        default="none",
        help="Optional feature blocks (A,B,C,O,P) or none.",
    )
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
    seeds = _parse_seeds(args.seeds)
    feature_blocks = _parse_feature_blocks(args.feature_blocks)
    params = CatBoostHyperParams(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=seeds[0],
    )

    metrics = train_baseline_cv_multiseed(
        train_csv_path=args.train_csv,
        models_dir=args.models_dir,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
        params=params,
        folds=args.folds,
        seeds=seeds,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
        feature_blocks=feature_blocks,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
