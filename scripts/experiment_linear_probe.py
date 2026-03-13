#!/usr/bin/env python3
"""Run a sparse linear probe and compare it with the incumbent blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import ID_COLUMN
from churn_baseline.diagnostics import load_merged_oof_matrix, parse_feature_blocks_arg, parse_oof_input_spec
from churn_baseline.linear_probe import (
    list_linear_probe_families,
    list_linear_probe_feature_modes,
    run_linear_probe_cv,
)
from churn_baseline.specialist import build_reference_prediction


def _parse_alpha_grid(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"alpha values must be in [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("alpha grid must contain at least one value")
    return values


def _load_weights(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Could not find weights in {path}")
    return {str(name): float(value) for name, value in weights.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sparse linear probe smoke test")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument(
        "--feature-blocks",
        default="H,R,S,V",
        help="Comma-separated feature blocks for the linear probe.",
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-family",
        default="logistic",
        choices=list(list_linear_probe_families()),
        help="Linear probe family to evaluate.",
    )
    parser.add_argument(
        "--feature-mode",
        default="raw",
        choices=list(list_linear_probe_feature_modes()),
        help="Feature mode: raw engineered matrix or compact teacher meta features.",
    )
    parser.add_argument("--c-value", type=float, default=0.5, help="Inverse regularization strength.")
    parser.add_argument("--max-iter", type=int, default=150, help="Maximum logistic iterations.")
    parser.add_argument("--tol", type=float, default=0.001, help="Optimization tolerance for logistic regression.")
    parser.add_argument("--spline-n-knots", type=int, default=6, help="Knots for spline_logistic numeric basis.")
    parser.add_argument("--spline-degree", type=int, default=3, help="Degree for spline_logistic numeric basis.")
    parser.add_argument("--tree-iterations", type=int, default=300, help="Boosting iterations for catboost_meta.")
    parser.add_argument("--tree-depth", type=int, default=4, help="Tree depth for catboost_meta.")
    parser.add_argument(
        "--tree-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for catboost_meta.",
    )
    parser.add_argument(
        "--tree-l2-leaf-reg",
        type=float,
        default=8.0,
        help="L2 leaf regularization for catboost_meta.",
    )
    parser.add_argument(
        "--tree-early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds for catboost_meta CV folds.",
    )
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--model-path",
        default="artifacts/models/linear_probe_smoke.pkl",
        help="Output pickle path for the full-train linear probe.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/linear_probe_smoke_metrics.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/linear_probe_smoke_oof.csv",
        help="Output OOF CSV path.",
    )
    parser.add_argument(
        "--reference-weights-json",
        default="",
        help="Optional weights JSON for incumbent OOF reference blend. Required by --feature-mode teacher_meta.",
    )
    parser.add_argument(
        "--reference-is-oof",
        action="store_true",
        help="Acknowledge that --reference-weights-json and --oof point to true OOF predictions aligned by id.",
    )
    parser.add_argument(
        "--oof",
        action="append",
        default=[],
        help=(
            "OOF input spec: <name>=<path>[#<prediction_column>]. "
            "Required whenever --reference-weights-json is provided; teacher_meta also needs pred_* teacher components."
        ),
    )
    parser.add_argument(
        "--alpha-grid",
        default="0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.50",
        help="Blend alphas for probe-vs-incumbent scan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_pred = None
    teacher_component_frame = None
    if args.reference_weights_json:
        if not args.reference_is_oof:
            raise ValueError("--reference-is-oof is required when --reference-weights-json is provided.")
        if not args.oof:
            raise ValueError("--oof is required when --reference-weights-json is provided.")
        specs = [parse_oof_input_spec(raw) for raw in args.oof]
        merged_oof, _ = load_merged_oof_matrix(specs)
        reference_pred = build_reference_prediction(merged_oof, _load_weights(args.reference_weights_json))
        pred_columns = [column for column in merged_oof.columns if column.startswith("pred_")]
        if args.feature_mode == "teacher_meta" and not pred_columns:
            raise ValueError("--feature-mode teacher_meta requires OOF inputs that merge into pred_* columns.")
        teacher_component_frame = merged_oof[[ID_COLUMN, *pred_columns]].copy()
    elif args.feature_mode == "teacher_meta":
        raise ValueError("--reference-weights-json and --oof are required when --feature-mode teacher_meta.")

    metrics = run_linear_probe_cv(
        train_csv_path=args.train_csv,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
        feature_blocks=parse_feature_blocks_arg(args.feature_blocks),
        folds=args.folds,
        random_state=args.random_state,
        c_value=args.c_value,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=args.verbose,
        model_family=args.model_family,
        feature_mode=args.feature_mode,
        spline_n_knots=args.spline_n_knots,
        spline_degree=args.spline_degree,
        reference_pred=reference_pred,
        teacher_component_frame=teacher_component_frame,
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
        reference_is_oof=bool(args.reference_weights_json and args.reference_is_oof),
        tree_iterations=args.tree_iterations,
        tree_depth=args.tree_depth,
        tree_learning_rate=args.tree_learning_rate,
        tree_l2_leaf_reg=args.tree_l2_leaf_reg,
        tree_early_stopping_rounds=args.tree_early_stopping_rounds,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
