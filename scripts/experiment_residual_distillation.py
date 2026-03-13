"""CLI for total residual distillation smoke against incumbent v3."""

from __future__ import annotations

import argparse
import json

from churn_baseline.config import CatBoostHyperParams
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.residual_distillation import ResidualDistillationConfig, run_total_residual_distillation_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the minimal total residual distillation smoke: learn "
            "v3 - base_reference as a regression target and compare directly against v3."
        )
    )
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Competition train CSV.")
    parser.add_argument("--out-dir", default="artifacts/reports", help="Directory for analysis and metrics.")
    parser.add_argument("--model-dir", default="artifacts/models", help="Directory for the CatBoostRegressor artifact.")
    parser.add_argument("--label", default="residual_distillation_smoke", help="Artifact prefix label.")
    parser.add_argument(
        "--feature-blocks",
        default="H,R,S,V",
        help="Comma-separated feature blocks for the distilled regressor.",
    )
    parser.add_argument(
        "--alpha-grid",
        default="0.25,0.5,1.0,2.0,4.0",
        help="Comma-separated alpha scan over base_reference + alpha * distilled_delta_pred.",
    )
    parser.add_argument("--folds", type=int, default=2, help="Number of CV folds for the smoke.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--iterations", type=int, default=150, help="CatBoostRegressor iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="CatBoostRegressor learning rate.")
    parser.add_argument("--depth", type=int, default=6, help="CatBoostRegressor depth.")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="CatBoostRegressor L2 regularization.")
    parser.add_argument("--early-stopping-rounds", type=int, default=60, help="Early stopping rounds.")
    parser.add_argument("--verbose", type=int, default=0, help="CatBoost verbosity.")
    return parser


def _parse_alpha_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in str(raw).split(",") if token.strip())
    if not values:
        raise ValueError("alpha-grid must contain at least one numeric value")
    return values


def main() -> None:
    args = build_parser().parse_args()
    feature_blocks = tuple(
        normalize_feature_blocks([token.strip() for token in str(args.feature_blocks).split(",") if token.strip()])
    )
    config = ResidualDistillationConfig(
        label=str(args.label),
        feature_blocks=feature_blocks,
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
    )
    params = CatBoostHyperParams(
        iterations=int(args.iterations),
        learning_rate=float(args.learning_rate),
        depth=int(args.depth),
        l2_leaf_reg=float(args.l2_leaf_reg),
        random_seed=int(args.random_state),
    )
    summary = run_total_residual_distillation_smoke(
        train_csv_path=args.train_csv,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        config=config,
        params=params,
        folds=int(args.folds),
        random_state=int(args.random_state),
        early_stopping_rounds=int(args.early_stopping_rounds),
        verbose=int(args.verbose),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
