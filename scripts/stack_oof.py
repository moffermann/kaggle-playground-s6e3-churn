#!/usr/bin/env python3
"""Train and evaluate a cross-fitted stacker on model OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.config import ID_COLUMN
from churn_baseline.oof_tools import (
    OOF_TARGET_COLUMN,
    compute_model_auc_table,
    load_baseline_auc,
    load_merged_oof,
    parse_oof_input,
    stack_oof_cross_fitted,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-fitted stacking on OOF predictions")
    parser.add_argument(
        "--oof",
        action="append",
        required=True,
        help="OOF input spec: <name>=<path>[#<prediction_column>]. Repeat for each model.",
    )
    parser.add_argument(
        "--stacker",
        default="logistic",
        choices=("logistic", "ridge"),
        help="Meta-model family for stacking.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Cross-fit folds for stacker.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for stacker CV.")
    parser.add_argument("--logistic-c", type=float, default=1.0, help="Inverse regularization (logistic).")
    parser.add_argument("--logistic-max-iter", type=int, default=2000, help="Max iterations (logistic).")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="L2 strength (ridge).")
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
        default="artifacts/reports/model_diversity_stack_oof.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--out-oof-csv",
        default="artifacts/reports/model_diversity_stack_oof.csv",
        help="Output stacked OOF predictions CSV path.",
    )
    parser.add_argument(
        "--stack-column",
        default="oof_stack",
        help="Column name for stacked OOF predictions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [parse_oof_input(raw) for raw in args.oof]
    merged, model_columns = load_merged_oof(inputs, id_column=ID_COLUMN, target_column=OOF_TARGET_COLUMN)
    auc_table = compute_model_auc_table(merged, model_columns, target_column=OOF_TARGET_COLUMN)

    stack_result = stack_oof_cross_fitted(
        merged,
        model_columns,
        stacker=args.stacker,
        folds=args.folds,
        random_state=args.random_state,
        logistic_c=args.logistic_c,
        logistic_max_iter=args.logistic_max_iter,
        ridge_alpha=args.ridge_alpha,
        target_column=OOF_TARGET_COLUMN,
    )

    baseline_auc = None
    baseline_metric = None
    if not args.skip_baseline:
        baseline_auc, baseline_metric = load_baseline_auc(args.baseline_metrics_path)

    oof_out = pd.DataFrame(
        {
            ID_COLUMN: merged[ID_COLUMN].values,
            OOF_TARGET_COLUMN: merged[OOF_TARGET_COLUMN].astype(int).values,
            args.stack_column: stack_result["oof_pred"],
        }
    )
    oof_path = Path(args.out_oof_csv)
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_out.to_csv(oof_path, index=False)

    best_base_auc = float(auc_table[0]["oof_auc"])
    summary = {
        "rows": int(len(merged)),
        "model_count": int(len(model_columns)),
        "id_column": ID_COLUMN,
        "target_column": OOF_TARGET_COLUMN,
        "model_columns": model_columns,
        "model_auc_table": auc_table,
        "stacker": stack_result["stacker"],
        "stack_oof_auc": float(stack_result["oof_auc"]),
        "stack_cv_mean_auc": float(stack_result["cv_mean_auc"]),
        "stack_cv_std_auc": float(stack_result["cv_std_auc"]),
        "stack_cv_fold_metrics": stack_result["cv_fold_metrics"],
        "stack_full_coefficients": stack_result["full_coefficients"],
        "stack_full_intercept": float(stack_result["full_intercept"]),
        "stack_params": stack_result["params"],
        "best_base_model_auc": best_base_auc,
        "delta_stack_vs_best_base_auc": float(stack_result["oof_auc"]) - best_base_auc,
        "baseline_auc": None if baseline_auc is None else float(baseline_auc),
        "baseline_auc_metric": baseline_metric,
        "baseline_metrics_path": None if args.skip_baseline else str(args.baseline_metrics_path),
        "delta_stack_vs_baseline_auc": (
            None if baseline_auc is None else float(stack_result["oof_auc"] - float(baseline_auc))
        ),
        "oof_output_path": str(oof_path),
        "stack_column": args.stack_column,
    }

    json_path = Path(args.out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(json_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
