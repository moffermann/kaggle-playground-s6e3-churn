#!/usr/bin/env python3
"""Analyze multiple OOF artifacts: per-model AUC and correlation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.config import ID_COLUMN
from churn_baseline.oof_tools import (
    OOF_TARGET_COLUMN,
    compute_model_auc_table,
    compute_oof_correlation,
    load_baseline_auc,
    load_merged_oof,
    parse_oof_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze OOF predictions from multiple models")
    parser.add_argument(
        "--oof",
        action="append",
        required=True,
        help="OOF input spec: <name>=<path>[#<prediction_column>]. Repeat for each model.",
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
        default="artifacts/reports/model_diversity_oof_analysis.json",
        help="Output analysis JSON path.",
    )
    parser.add_argument(
        "--out-corr-csv",
        default="artifacts/reports/model_diversity_oof_correlation.csv",
        help="Output correlation CSV path.",
    )
    parser.add_argument(
        "--out-merged-csv",
        default="",
        help="Optional path to persist merged OOF matrix (id/target/pred_*).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    oof_inputs = [parse_oof_input(raw) for raw in args.oof]
    merged, model_columns = load_merged_oof(oof_inputs, id_column=ID_COLUMN, target_column=OOF_TARGET_COLUMN)

    auc_table = compute_model_auc_table(merged, model_columns, target_column=OOF_TARGET_COLUMN)
    corr_df = compute_oof_correlation(merged, model_columns)

    baseline_auc = None
    baseline_key = None
    if not args.skip_baseline:
        baseline_auc, baseline_key = load_baseline_auc(args.baseline_metrics_path)
        for row in auc_table:
            row["delta_vs_baseline_auc"] = float(row["oof_auc"] - baseline_auc)
            row["baseline_auc"] = float(baseline_auc)
            row["baseline_auc_metric"] = baseline_key

    best_model = auc_table[0]["model_column"]
    summary = {
        "model_count": int(len(model_columns)),
        "rows": int(len(merged)),
        "id_column": ID_COLUMN,
        "target_column": OOF_TARGET_COLUMN,
        "model_columns": model_columns,
        "best_model_column": str(best_model),
        "baseline_auc": float(baseline_auc) if baseline_auc is not None else None,
        "baseline_auc_metric": baseline_key,
        "baseline_metrics_path": None if args.skip_baseline else str(args.baseline_metrics_path),
        "auc_table": auc_table,
    }
    write_json(args.out_json, summary)

    corr_out_path = Path(args.out_corr_csv)
    corr_out_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(corr_out_path, index=True)

    if args.out_merged_csv:
        merged_out_path = Path(args.out_merged_csv)
        merged_out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_out_path, index=False)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
