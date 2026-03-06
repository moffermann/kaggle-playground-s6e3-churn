#!/usr/bin/env python3
"""Evaluate and optimize OOF blends from multiple model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.config import ID_COLUMN
from churn_baseline.oof_tools import (
    OOF_TARGET_COLUMN,
    blend_auc,
    blend_predictions,
    compute_model_auc_table,
    grid_search_blend_weights,
    load_baseline_auc,
    load_merged_oof,
    normalize_weights,
    optimize_blend_coordinate_descent,
    parse_oof_input,
)


def _parse_manual_weights(raw: str, model_columns: list[str]) -> np.ndarray:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Manual weights are empty.")

    mapping: dict[str, float] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"Invalid weight token '{token}'. Expected format pred_name=value")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        mapping[key] = float(value)

    missing = [column for column in model_columns if column not in mapping]
    if missing:
        raise ValueError(f"Manual weights missing model columns: {missing}")
    extras = [key for key in mapping if key not in model_columns]
    if extras:
        raise ValueError(f"Manual weights include unknown model columns: {extras}")

    return normalize_weights(np.array([mapping[column] for column in model_columns], dtype="float64"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend OOF predictions from multiple models")
    parser.add_argument(
        "--oof",
        action="append",
        required=True,
        help="OOF input spec: <name>=<path>[#<prediction_column>]. Repeat for each model.",
    )
    parser.add_argument(
        "--method",
        choices=("equal", "manual", "coordinate", "grid"),
        default="coordinate",
        help="Blend optimization method.",
    )
    parser.add_argument(
        "--manual-weights",
        default="",
        help="Manual weights as comma-separated pred_column=value (requires --method manual).",
    )
    parser.add_argument(
        "--coordinate-step",
        type=float,
        default=0.02,
        help="Step size for coordinate-descent optimization.",
    )
    parser.add_argument(
        "--coordinate-rounds",
        type=int,
        default=25,
        help="Max optimization rounds for coordinate descent.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.05,
        help="Simplex grid step for --method grid.",
    )
    parser.add_argument(
        "--grid-top-k",
        type=int,
        default=5,
        help="Top K weight combinations to keep for --method grid.",
    )
    parser.add_argument(
        "--baseline-metrics-path",
        default="artifacts/reports/train_cv_multiseed_full_hiiter_metrics.json",
        help="Optional baseline metrics JSON with ensemble_oof_auc/oof_auc/holdout_auc.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not load/compare against baseline metrics.",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/model_diversity_blend_metrics.json",
        help="Output blend metrics JSON path.",
    )
    parser.add_argument(
        "--out-oof-csv",
        default="artifacts/reports/model_diversity_blend_oof.csv",
        help="Output blended OOF CSV path.",
    )
    parser.add_argument(
        "--blend-column",
        default="oof_blend",
        help="Output column name for blended prediction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.method == "manual" and not args.manual_weights.strip():
        raise ValueError("--method manual requires --manual-weights")

    merged, model_columns = load_merged_oof(
        [parse_oof_input(raw) for raw in args.oof],
        id_column=ID_COLUMN,
        target_column=OOF_TARGET_COLUMN,
    )

    equal_weights = np.full(shape=len(model_columns), fill_value=1.0 / float(len(model_columns)))
    equal_auc = blend_auc(merged, model_columns, equal_weights, target_column=OOF_TARGET_COLUMN)

    best_result: dict[str, object]
    top_grid: list[dict[str, object]] = []
    if args.method == "equal":
        best_result = {
            "auc": float(equal_auc),
            "weights": {model_columns[idx]: float(equal_weights[idx]) for idx in range(len(model_columns))},
            "method": "equal",
        }
    elif args.method == "manual":
        manual_weights = _parse_manual_weights(args.manual_weights, model_columns)
        manual_auc = blend_auc(merged, model_columns, manual_weights, target_column=OOF_TARGET_COLUMN)
        best_result = {
            "auc": float(manual_auc),
            "weights": {model_columns[idx]: float(manual_weights[idx]) for idx in range(len(model_columns))},
            "method": "manual",
        }
    elif args.method == "grid":
        top_grid = grid_search_blend_weights(
            merged,
            model_columns,
            grid_step=args.grid_step,
            top_k=args.grid_top_k,
            target_column=OOF_TARGET_COLUMN,
        )
        best_grid = top_grid[0]
        best_result = {
            "auc": float(best_grid["auc"]),
            "weights": dict(best_grid["weights"]),
            "method": "grid",
            "top_grid": top_grid,
        }
    else:
        coordinate = optimize_blend_coordinate_descent(
            merged,
            model_columns,
            step=args.coordinate_step,
            max_rounds=args.coordinate_rounds,
            target_column=OOF_TARGET_COLUMN,
        )
        best_result = {
            "auc": float(coordinate["auc"]),
            "weights": dict(coordinate["weights"]),
            "method": "coordinate",
            "rounds_used": int(coordinate["rounds_used"]),
        }

    selected_weights = np.array(
        [float(best_result["weights"][column]) for column in model_columns], dtype="float64"
    )
    blended_pred = blend_predictions(merged, model_columns, selected_weights)

    output_oof = merged.copy()
    output_oof[args.blend_column] = blended_pred
    oof_out_path = Path(args.out_oof_csv)
    oof_out_path.parent.mkdir(parents=True, exist_ok=True)
    output_oof.to_csv(oof_out_path, index=False)

    model_auc_table = compute_model_auc_table(merged, model_columns, target_column=OOF_TARGET_COLUMN)
    baseline_auc = None
    baseline_key = None
    if not args.skip_baseline:
        baseline_auc, baseline_key = load_baseline_auc(args.baseline_metrics_path)

    report = {
        "rows": int(len(merged)),
        "model_count": int(len(model_columns)),
        "id_column": ID_COLUMN,
        "target_column": OOF_TARGET_COLUMN,
        "model_columns": model_columns,
        "model_auc_table": model_auc_table,
        "equal_blend_auc": float(equal_auc),
        "selected_method": args.method,
        "selected_blend_auc": float(best_result["auc"]),
        "selected_weights": dict(best_result["weights"]),
        "baseline_auc": float(baseline_auc) if baseline_auc is not None else None,
        "baseline_auc_metric": baseline_key,
        "baseline_metrics_path": None if args.skip_baseline else str(args.baseline_metrics_path),
        "delta_selected_vs_equal_auc": float(best_result["auc"] - equal_auc),
        "delta_selected_vs_baseline_auc": (
            None if baseline_auc is None else float(best_result["auc"] - baseline_auc)
        ),
        "oof_output_path": str(oof_out_path),
        "blend_column": str(args.blend_column),
    }
    if "rounds_used" in best_result:
        report["coordinate_rounds_used"] = int(best_result["rounds_used"])
    if "top_grid" in best_result:
        report["grid_top_results"] = list(best_result["top_grid"])

    write_json(args.out_json, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
