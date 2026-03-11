#!/usr/bin/env python3
"""Run feature engineering experiments with CV and optional baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import CatBoostHyperParams
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.pipeline import train_baseline_cv


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def _load_baseline_auc(metrics_path: str | Path) -> tuple[float, str]:
    with Path(metrics_path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    for key in ("ensemble_oof_auc", "oof_auc", "holdout_auc"):
        if key in payload:
            return float(payload[key]), key
    raise ValueError(
        f"Could not find baseline AUC in {metrics_path}. "
        "Expected one of: ensemble_oof_auc, oof_auc, holdout_auc."
    )


def _normalize_monotonic_option(raw: str) -> str:
    value = str(raw or "none").strip().lower()
    if value in {"", "none", "off", "false", "0"}:
        return "none"
    if value != "minimal":
        raise ValueError(f"Unsupported monotonic option '{raw}'. Supported: none, minimal.")
    return value


def _monotonic_constraints_from_preset(preset: str) -> dict[str, int] | None:
    normalized = _normalize_monotonic_option(preset)
    if normalized == "none":
        return None
    return {
        "tenure": -1,
        "payment_friction_index": 1,
        "contract_commitment_ordinal": -1,
        "is_manual_payment": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CV experiment with feature blocks")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--feature-blocks",
        default="A",
        help="Comma-separated feature blocks to enable (supported: A,B,C,F,G,T,H,R,S,V,O,P). Use 'none' for baseline.",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/models/fe_experiment_cv.cbm",
        help="Output model path",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/fe_experiment_cv_metrics.json",
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/fe_experiment_cv_oof.csv",
        help="Output OOF predictions CSV path",
    )
    parser.add_argument(
        "--baseline-metrics-path",
        default="artifacts/reports/train_cv_multiseed_full_hiiter_metrics.json",
        help="JSON path with incumbent metric (ensemble_oof_auc/oof_auc/holdout_auc)",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of Stratified K folds")
    parser.add_argument(
        "--stratify-mode",
        choices=("target", "composite"),
        default="target",
        help="CV stratification: target only, or target plus family fallback.",
    )
    parser.add_argument(
        "--monotonic-feature-set",
        default="none",
        help="Optional monotonic helper features to append (supported: none, minimal).",
    )
    parser.add_argument(
        "--monotonic-preset",
        default="none",
        help="Optional CatBoost monotonic preset (supported: none, minimal).",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--iterations", type=int, default=2200, help="Max boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="L2 regularization")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=120,
        help="Early stopping rounds per fold",
    )
    parser.add_argument("--verbose", type=int, default=200, help="CatBoost verbose frequency")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_blocks = _parse_feature_blocks(args.feature_blocks)
    monotonic_feature_set = _normalize_monotonic_option(args.monotonic_feature_set)
    monotonic_preset = _normalize_monotonic_option(args.monotonic_preset)
    if monotonic_feature_set == "none" and monotonic_preset != "none":
        monotonic_feature_set = monotonic_preset
    include_monotonic_features = monotonic_feature_set != "none"
    monotonic_constraints = _monotonic_constraints_from_preset(monotonic_preset)
    params = CatBoostHyperParams(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
        monotone_constraints=monotonic_constraints,
    )

    metrics = train_baseline_cv(
        train_csv_path=args.train_csv,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
        params=params,
        folds=args.folds,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
        feature_blocks=feature_blocks,
        stratify_mode=args.stratify_mode,
        include_monotonic_features=include_monotonic_features,
    )

    baseline_auc, baseline_key = _load_baseline_auc(args.baseline_metrics_path)
    metrics["baseline_auc"] = baseline_auc
    metrics["baseline_auc_metric"] = baseline_key
    metrics["baseline_metrics_path"] = args.baseline_metrics_path
    metrics["delta_vs_baseline_auc"] = float(metrics["oof_auc"] - baseline_auc)
    metrics["experiment_name"] = f"blocks_{'_'.join(feature_blocks) if feature_blocks else 'none'}"
    metrics["monotonic_feature_set"] = monotonic_feature_set
    metrics["monotonic_preset"] = monotonic_preset

    with Path(args.metrics_path).open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
