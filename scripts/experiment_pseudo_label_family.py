#!/usr/bin/env python3
"""Run offline family-level pseudo-labeling experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import CatBoostHyperParams, ID_COLUMN
from churn_baseline.diagnostics import load_merged_oof_matrix, parse_oof_input_spec
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.pseudo_labeling import (
    DEFAULT_FAMILY_KEY,
    list_supported_label_modes,
    run_family_pseudo_label_experiment,
)
from churn_baseline.specialist import build_reference_prediction


DEFAULT_OOF_SPECS = (
    "cb=artifacts/reports/train_cv_multiseed_gate_s5_hiiter_oof.csv#oof_ensemble",
    "xgb=artifacts/reports/train_xgboost_cv_multiseed_hiiter_oof.csv#oof_ensemble",
    "lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred",
    "r=artifacts/reports/fe_blockR_hiiter_oof.csv#oof_pred",
    "rv=artifacts/reports/fe_blockRV_hiiter_oof.csv#oof_pred",
)


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def _parse_float_grid(raw: str, *, allow_zero: bool = False, upper_inclusive: float | None = None) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if allow_zero:
            if value < 0.0:
                raise ValueError(f"Grid values must be >= 0, got {value}")
        else:
            if value <= 0.0:
                raise ValueError(f"Grid values must be > 0, got {value}")
        if upper_inclusive is not None and value > upper_inclusive:
            raise ValueError(f"Grid values must be <= {upper_inclusive}, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Grid must contain at least one value")
    return values


def _load_weights(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Could not find weights in {path}")
    return {str(name): float(value) for name, value in weights.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline pseudo-labeling by family")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--family-key",
        default=DEFAULT_FAMILY_KEY,
        help="Exact segment5 family key: PaymentMethod__Contract__InternetService__PaperlessBilling__tenure_bin",
    )
    parser.add_argument(
        "--feature-blocks",
        default="H,R,S,V",
        help="Comma-separated feature blocks for the student model.",
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
        help="Weights JSON for the teacher blend used to pseudo-label train rows.",
    )
    parser.add_argument(
        "--holdout-size",
        type=float,
        default=0.15,
        help="Global offline holdout fraction.",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.15,
        help="Inner validation fraction used for early stopping.",
    )
    parser.add_argument(
        "--pseudo-pool-fraction",
        type=float,
        default=0.50,
        help="Fraction of target family rows hidden as pseudo pool inside each repeat.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated splits.")
    parser.add_argument("--random-state", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--upper-thresholds",
        default="0.995,0.99,0.98",
        help="Comma-separated positive pseudo-label thresholds.",
    )
    parser.add_argument(
        "--lower-thresholds",
        default="0.005,0.01,0.02",
        help="Comma-separated negative pseudo-label thresholds.",
    )
    parser.add_argument(
        "--pseudo-weights",
        default="0.05,0.10,0.20,0.35,0.50",
        help="Comma-separated pseudo-row sample weights.",
    )
    parser.add_argument(
        "--label-mode",
        default="hard",
        choices=list(list_supported_label_modes()),
        help="Pseudo-label target mode: hard thresholded labels or soft duplicated labels.",
    )
    parser.add_argument(
        "--scale-pseudo-weight-by-confidence",
        action="store_true",
        help="Scale each pseudo-row weight by teacher confidence |p-0.5|*2.",
    )
    parser.add_argument(
        "--require-teacher-agreement",
        action="store_true",
        help="Require all teacher components to agree on the pseudo-label side of 0.5.",
    )
    parser.add_argument(
        "--max-teacher-std",
        type=float,
        default=None,
        help="Optional maximum std across teacher components for selected pseudo rows.",
    )
    parser.add_argument(
        "--min-selected-rows",
        type=int,
        default=100,
        help="Minimum selected pseudo rows per config/repeat before training.",
    )
    parser.add_argument("--iterations", type=int, default=150, help="Max boosting rounds.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth.")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="L2 regularization.")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=40,
        help="Early stopping rounds.",
    )
    parser.add_argument("--verbose", type=int, default=0, help="CatBoost verbose frequency.")
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/pseudo_label_family_smoke_metrics.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--results-csv-path",
        default="artifacts/reports/pseudo_label_family_smoke_results.csv",
        help="Output detailed results CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_blocks = _parse_feature_blocks(args.feature_blocks)
    upper_thresholds = _parse_float_grid(args.upper_thresholds, upper_inclusive=1.0)
    lower_thresholds = _parse_float_grid(args.lower_thresholds, allow_zero=True, upper_inclusive=0.5)
    pseudo_weights = _parse_float_grid(args.pseudo_weights, upper_inclusive=1.0)
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
    reference_component_frame = None
    if component_columns:
        reference_component_frame = merged_oof[[ID_COLUMN, *component_columns]].copy()

    metrics = run_family_pseudo_label_experiment(
        train_csv_path=args.train_csv,
        family_key=args.family_key,
        reference_pred=reference_pred,
        reference_component_frame=reference_component_frame,
        feature_blocks=feature_blocks,
        params=params,
        holdout_size=args.holdout_size,
        valid_size=args.valid_size,
        pseudo_pool_fraction=args.pseudo_pool_fraction,
        repeats=args.repeats,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
        upper_thresholds=upper_thresholds,
        lower_thresholds=lower_thresholds,
        pseudo_weights=pseudo_weights,
        label_mode=args.label_mode,
        scale_weight_by_confidence=args.scale_pseudo_weight_by_confidence,
        require_teacher_agreement=args.require_teacher_agreement,
        max_teacher_std=args.max_teacher_std,
        min_selected_rows=args.min_selected_rows,
        metrics_path=args.metrics_path,
        results_csv_path=args.results_csv_path,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
