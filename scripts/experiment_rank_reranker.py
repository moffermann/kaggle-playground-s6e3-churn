#!/usr/bin/env python3
"""Train and evaluate a ranking-aware teacher reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import CatBoostHyperParams, ID_COLUMN
from churn_baseline.diagnostics import load_merged_oof_matrix, parse_feature_blocks_arg, parse_oof_input_spec
from churn_baseline.pipeline import SUPPORTED_STRATIFY_MODES
from churn_baseline.rank_reranker import list_rank_reranker_losses, run_rank_reranker_cv
from churn_baseline.specialist import build_reference_prediction, list_specialist_presets


DEFAULT_OOF_SPECS = (
    "cb=artifacts/reports/train_cv_multiseed_gate_s5_hiiter_oof.csv#oof_ensemble",
    "xgb=artifacts/reports/train_xgboost_cv_multiseed_hiiter_oof.csv#oof_ensemble",
    "lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred",
    "r=artifacts/reports/fe_blockR_hiiter_oof.csv#oof_pred",
    "rv=artifacts/reports/fe_blockRV_hiiter_oof.csv#oof_pred",
)


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
    parser = argparse.ArgumentParser(description="Run ranking-aware teacher reranker experiment")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--preset",
        default="global",
        choices=["global", *sorted(list_specialist_presets().keys())],
        help="Optional local rerank mask. Use 'global' to rerank every row.",
    )
    parser.add_argument(
        "--feature-blocks",
        default="H,R,S,V",
        help="Comma-separated feature blocks for the ranker (supported: A,B,C,G,H,R,S,V,O,P).",
    )
    parser.add_argument(
        "--oof",
        action="append",
        default=[],
        help="OOF input spec: <name>=<path>[#<prediction_column>]. Defaults to incumbent cb/xgb/lgb/r/rv.",
    )
    parser.add_argument(
        "--reference-weights-json",
        default="artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json",
        help="Weights JSON for the incumbent blend.",
    )
    parser.add_argument(
        "--loss-function",
        default="PairLogitPairwise",
        choices=list(list_rank_reranker_losses()),
        help="CatBoostRanker loss function.",
    )
    parser.add_argument(
        "--alpha-grid",
        default="0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00",
        help="Comma-separated alpha values for rank-blend scan.",
    )
    parser.add_argument(
        "--stratify-mode",
        default="target",
        choices=list(SUPPORTED_STRATIFY_MODES),
        help="Outer CV stratification mode.",
    )
    parser.add_argument("--min-query-rows", type=int, default=50, help="Minimum rows per query level.")
    parser.add_argument("--min-query-positive-rows", type=int, default=3, help="Minimum positive rows per query.")
    parser.add_argument("--min-query-negative-rows", type=int, default=10, help="Minimum negative rows per query.")
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=1000,
        help="Cap sampled positive-vs-negative pairs per query group.",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/models/rank_reranker_smoke.cbm",
        help="Output ranker model path.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/rank_reranker_smoke_metrics.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/rank_reranker_smoke_oof.csv",
        help="Output OOF CSV path.",
    )
    parser.add_argument("--folds", type=int, default=2, help="Number of CV folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--iterations", type=int, default=150, help="Max boosting rounds.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth.")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="L2 regularization.")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=40,
        help="Early stopping rounds for the ranker.",
    )
    parser.add_argument("--verbose", type=int, default=0, help="CatBoost verbose frequency.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_blocks = parse_feature_blocks_arg(args.feature_blocks)
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    params = CatBoostHyperParams(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
    )

    raw_specs = args.oof if args.oof else list(DEFAULT_OOF_SPECS)
    specs = [parse_oof_input_spec(raw) for raw in raw_specs]
    merged_oof, _ = load_merged_oof_matrix(specs)
    reference_pred = build_reference_prediction(merged_oof, _load_weights(args.reference_weights_json))
    component_columns = [column for column in merged_oof.columns if column.startswith("pred_")]
    reference_component_frame = merged_oof[[ID_COLUMN, *component_columns]].copy()

    metrics = run_rank_reranker_cv(
        train_csv_path=args.train_csv,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
        params=params,
        feature_blocks=feature_blocks,
        reference_pred=reference_pred,
        reference_component_frame=reference_component_frame,
        preset=None if args.preset == "global" else args.preset,
        folds=args.folds,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
        loss_function=args.loss_function,
        alpha_grid=alpha_grid,
        stratify_mode=args.stratify_mode,
        min_query_rows=args.min_query_rows,
        min_query_positive_rows=args.min_query_positive_rows,
        min_query_negative_rows=args.min_query_negative_rows,
        max_pairs_per_group=args.max_pairs_per_group,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
