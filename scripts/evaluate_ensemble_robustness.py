#!/usr/bin/env python3
"""Evaluate ensemble robustness with repeated split validation on OOF matrix."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.diagnostics import (
    OOF_TARGET_COLUMN,
    evaluate_ensemble_robustness,
    load_json_if_exists,
    load_merged_oof_matrix,
    parse_oof_input_spec,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate equal/rank/weighted ensemble robustness")
    parser.add_argument(
        "--oof",
        action="append",
        required=True,
        help="OOF input spec: <name>=<path>[#<prediction_column>]. Repeat for each model.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated CV runs.")
    parser.add_argument("--folds", type=int, default=5, help="Folds per repeat.")
    parser.add_argument("--random-state", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--weighted-step",
        type=float,
        default=0.02,
        help="Coordinate descent step for weighted blend.",
    )
    parser.add_argument(
        "--weighted-rounds",
        type=int,
        default=25,
        help="Max coordinate descent rounds for weighted blend.",
    )
    parser.add_argument(
        "--baseline-metrics-path",
        default="artifacts/reports/train_cv_multiseed_full_hiiter_metrics.json",
        help="Optional baseline metrics JSON for delta reporting.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not load baseline metrics.",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/diagnostic_ensemble_robustness.json",
        help="Output summary JSON path.",
    )
    return parser.parse_args()


def _load_baseline_auc(metrics_path: str) -> tuple[float, str] | None:
    payload = load_json_if_exists(metrics_path)
    if payload is None:
        return None
    for key in ("ensemble_oof_auc", "oof_auc", "holdout_auc"):
        value = payload.get(key)
        if value is not None:
            return float(value), key
    return None


def main() -> int:
    args = parse_args()
    specs = [parse_oof_input_spec(raw) for raw in args.oof]
    merged, model_columns = load_merged_oof_matrix(specs, target_column=OOF_TARGET_COLUMN)

    results = evaluate_ensemble_robustness(
        merged,
        model_columns,
        target_column=OOF_TARGET_COLUMN,
        repeats=args.repeats,
        folds=args.folds,
        random_state=args.random_state,
        weighted_step=args.weighted_step,
        weighted_rounds=args.weighted_rounds,
    )

    baseline_auc = None
    baseline_key = None
    if not args.skip_baseline:
        baseline = _load_baseline_auc(args.baseline_metrics_path)
        if baseline is not None:
            baseline_auc, baseline_key = baseline

    if baseline_auc is not None:
        for method_name, method_payload in results["methods"].items():
            method_payload["delta_full_vs_baseline_auc"] = float(
                method_payload["full_auc"] - baseline_auc
            )
            method_payload["delta_cv_mean_vs_baseline_auc"] = float(
                method_payload["cv_mean_auc"] - baseline_auc
            )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "inputs": {
            "oof": args.oof,
            "repeats": int(args.repeats),
            "folds": int(args.folds),
            "random_state": int(args.random_state),
            "weighted_step": float(args.weighted_step),
            "weighted_rounds": int(args.weighted_rounds),
            "baseline_metrics_path": None if args.skip_baseline else args.baseline_metrics_path,
        },
        "baseline_auc": baseline_auc,
        "baseline_auc_metric": baseline_key,
        **results,
    }
    write_json(args.out_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
