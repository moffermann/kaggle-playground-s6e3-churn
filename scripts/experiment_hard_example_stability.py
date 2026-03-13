#!/usr/bin/env python3
"""Run hard-example stability smoke directly against incumbent v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import CatBoostHyperParams, ID_COLUMN
from churn_baseline.diagnostics import DEFAULT_REFERENCE_OOF_SPECS, load_merged_oof_matrix, load_reference_prediction_frame, parse_oof_input_spec
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.hard_example_stability import run_hard_example_stability_reranker_cv
from churn_baseline.uncertainty_band import DEFAULT_V3_OOF_SPEC
from churn_baseline.validation_protocol import DOMINANT_MACROFAMILY, SUPPORTED_STAGES, evaluate_validation_protocol


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def _parse_alpha_grid(raw: str) -> list[float]:
    values: list[float] = []
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hard-example stability reranker directly against v3")
    parser.add_argument("--stage", choices=SUPPORTED_STAGES, default="smoke")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument(
        "--reference-v3-oof",
        default=DEFAULT_V3_OOF_SPEC,
        help="Direct OOF spec for incumbent v3 (<path>[#prediction_column]).",
    )
    parser.add_argument(
        "--oof",
        action="append",
        default=[],
        help="Teacher component OOF input spec: <name>=<path>[#<prediction_column>]. Defaults to cb/xgb/lgb/r/rv.",
    )
    parser.add_argument(
        "--stability-feature-blocks",
        default="R,V",
        help="Comma-separated feature blocks used to generate repeated stability OOFs.",
    )
    parser.add_argument(
        "--reranker-feature-blocks",
        default="H,R,S,V",
        help="Comma-separated feature blocks used by the local residual reranker.",
    )
    parser.add_argument(
        "--family-level",
        choices=["segment3", "segment5"],
        default="segment3",
    )
    parser.add_argument(
        "--family-value",
        default=DOMINANT_MACROFAMILY,
        help="Family key used to constrain the hard-example mask.",
    )
    parser.add_argument(
        "--hard-score-quantile",
        type=float,
        default=0.80,
        help="Keep only rows above this within-family hard-example-score quantile.",
    )
    parser.add_argument(
        "--reference-band-half-width",
        type=float,
        default=None,
        help="Optional additional gate: abs(v3 - 0.5) <= width.",
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--stability-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--stability-iterations", type=int, default=120)
    parser.add_argument("--stability-learning-rate", type=float, default=0.05)
    parser.add_argument("--stability-depth", type=int, default=4)
    parser.add_argument("--stability-l2-leaf-reg", type=float, default=8.0)
    parser.add_argument("--reranker-iterations", type=int, default=180)
    parser.add_argument("--reranker-learning-rate", type=float, default=0.04)
    parser.add_argument("--reranker-depth", type=int, default=5)
    parser.add_argument("--reranker-l2-leaf-reg", type=float, default=8.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--alpha-grid",
        default="0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.50",
        help="Comma-separated alphas for local residual injection.",
    )
    parser.add_argument("--model-path", default="artifacts/models/hard_example_stability_smoke.cbm")
    parser.add_argument("--metrics-path", default="artifacts/reports/hard_example_stability_smoke_metrics.json")
    parser.add_argument("--oof-path", default="artifacts/reports/hard_example_stability_smoke_oof.csv")
    parser.add_argument(
        "--gate-path",
        default="artifacts/reports/validation_protocol_hard_example_stability_smoke_vs_v3.json",
        help="Validation reset report written after training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    stability_blocks = _parse_feature_blocks(args.stability_feature_blocks)
    reranker_blocks = _parse_feature_blocks(args.reranker_feature_blocks)

    reference_frame, reference_source = load_reference_prediction_frame(
        reference_oof_spec=args.reference_v3_oof,
        id_column=ID_COLUMN,
        target_column="target",
    )

    raw_specs = args.oof if args.oof else list(DEFAULT_REFERENCE_OOF_SPECS)
    specs = [parse_oof_input_spec(raw) for raw in raw_specs]
    merged_oof, _ = load_merged_oof_matrix(specs, id_column=ID_COLUMN, target_column="target")
    component_columns = [column for column in merged_oof.columns if column.startswith("pred_")]
    reference_component_frame = merged_oof[[ID_COLUMN, *component_columns]].copy()

    stability_params = CatBoostHyperParams(
        iterations=args.stability_iterations,
        learning_rate=args.stability_learning_rate,
        depth=args.stability_depth,
        l2_leaf_reg=args.stability_l2_leaf_reg,
        random_seed=args.random_state,
    )
    reranker_params = CatBoostHyperParams(
        iterations=args.reranker_iterations,
        learning_rate=args.reranker_learning_rate,
        depth=args.reranker_depth,
        l2_leaf_reg=args.reranker_l2_leaf_reg,
        random_seed=args.random_state,
    )

    metrics = run_hard_example_stability_reranker_cv(
        train_csv_path=args.train_csv,
        reference_frame=reference_frame,
        reference_component_frame=reference_component_frame,
        stability_feature_blocks=stability_blocks,
        stability_params=stability_params,
        stability_repeats=args.stability_repeats,
        reranker_feature_blocks=reranker_blocks,
        reranker_params=reranker_params,
        folds=args.folds,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
        alpha_grid=alpha_grid,
        family_level=args.family_level,
        family_value=args.family_value,
        hard_score_quantile=args.hard_score_quantile,
        reference_band_half_width=args.reference_band_half_width,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
    )

    gate = evaluate_validation_protocol(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        stage=args.stage,
        analysis_oof_path=args.oof_path,
        target_family_level=args.family_level,
        target_family_value=args.family_value,
        dominant_family_value=DOMINANT_MACROFAMILY,
        candidate_metrics_json=args.metrics_path,
        out_json_path=args.gate_path,
    )

    summary = {
        "metrics_path": str(Path(args.metrics_path)),
        "oof_path": str(Path(args.oof_path)),
        "gate_path": str(Path(args.gate_path)),
        "reference_source": reference_source,
        "component_specs": raw_specs,
        "metrics": metrics,
        "gate": gate,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
