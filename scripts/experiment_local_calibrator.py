#!/usr/bin/env python3
"""Run a local calibrator over the incumbent blend inside a hard cohort preset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.diagnostics import load_merged_oof_matrix, parse_oof_input_spec
from churn_baseline.specialist import (
    build_reference_prediction,
    list_calibration_methods,
    list_specialist_presets,
    run_local_calibrator_cv,
)


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
    parser = argparse.ArgumentParser(description="Run local calibrator experiment")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--preset",
        default="early_manual_internet",
        choices=sorted(list_specialist_presets().keys()),
        help="Hard cohort preset to calibrate.",
    )
    parser.add_argument(
        "--method",
        default="platt",
        choices=list(list_calibration_methods()),
        help="Local calibration method.",
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
        "--alpha-grid",
        default="0.25,0.50,0.75,1.00",
        help="Comma-separated alpha values for local override scan.",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/models/local_calibrator_smoke.pkl",
        help="Output calibrator bundle path.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/local_calibrator_smoke_metrics.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/local_calibrator_smoke_oof.csv",
        help="Output OOF CSV path.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of Stratified K folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alpha_grid = _parse_alpha_grid(args.alpha_grid)

    raw_specs = args.oof if args.oof else list(DEFAULT_OOF_SPECS)
    specs = [parse_oof_input_spec(raw) for raw in raw_specs]
    merged_oof, _ = load_merged_oof_matrix(specs)
    reference_pred = build_reference_prediction(merged_oof, _load_weights(args.reference_weights_json))

    metrics = run_local_calibrator_cv(
        train_csv_path=args.train_csv,
        preset=args.preset,
        reference_pred=reference_pred,
        method=args.method,
        folds=args.folds,
        random_state=args.random_state,
        alpha_grid=alpha_grid,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
