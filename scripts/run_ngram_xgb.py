#!/usr/bin/env python3
"""Run the minimal bi-gram target-encoding XGBoost experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import ensure_parent_dir
from churn_baseline.ngram_xgb import DEFAULT_NGRAM_BASE_COLUMNS, NgramXgbParams, train_ngram_xgb_cv


def _parse_columns(raw: str) -> list[str] | None:
    value = str(raw or "").strip()
    if not value:
        return None
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal ngram+TE+XGBoost CV")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--original-csv", default="", help="Optional original Telco CSV used only for TE support.")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoothing-alpha", type=float, default=20.0)
    parser.add_argument("--ngram-base-columns", default=",".join(DEFAULT_NGRAM_BASE_COLUMNS))
    parser.add_argument("--include-trigrams", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-child-weight", type=float, default=4.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.6)
    parser.add_argument("--reg-alpha", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--verbose-eval", type=int, default=0)
    parser.add_argument("--metrics-path", default="artifacts/reports/ngram_te_xgb_smoke_metrics.json")
    parser.add_argument("--oof-path", default="artifacts/reports/ngram_te_xgb_smoke_oof.csv")
    parser.add_argument("--test-pred-path", default="", help="Optional test prediction CSV path.")
    parser.add_argument(
        "--reference-v3-oof",
        default="",
        help="Optional v3 OOF CSV with columns id,target,candidate_pred to build analysis-oof directly.",
    )
    parser.add_argument(
        "--analysis-oof-path",
        default="",
        help="Optional merged analysis OOF path with id,target,reference_pred,candidate_pred.",
    )
    return parser.parse_args()


def _build_analysis_oof(
    *,
    reference_v3_oof_path: str | Path,
    candidate_oof_path: str | Path,
    analysis_oof_path: str | Path,
) -> str:
    reference = pd.read_csv(reference_v3_oof_path)[["id", "target", "candidate_pred"]].rename(
        columns={"candidate_pred": "reference_pred"}
    )
    candidate = pd.read_csv(candidate_oof_path)[["id", "target", "oof_pred"]].rename(columns={"oof_pred": "candidate_pred"})
    merged = reference.merge(candidate, on=["id", "target"], how="inner", validate="one_to_one")
    out_path = ensure_parent_dir(analysis_oof_path)
    merged.to_csv(out_path, index=False)
    return str(out_path)


def main() -> int:
    args = parse_args()
    params = NgramXgbParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    metrics = train_ngram_xgb_cv(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        original_csv_path=args.original_csv.strip() or None,
        params=params,
        folds=args.folds,
        inner_folds=args.inner_folds,
        random_state=args.random_state,
        smoothing_alpha=args.smoothing_alpha,
        include_trigrams=bool(args.include_trigrams),
        ngram_base_columns=_parse_columns(args.ngram_base_columns),
        verbose_eval=max(int(args.verbose_eval), 0),
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
        test_pred_path=args.test_pred_path.strip() or None,
    )
    if args.reference_v3_oof.strip() and args.analysis_oof_path.strip():
        metrics["analysis_oof_path"] = _build_analysis_oof(
            reference_v3_oof_path=args.reference_v3_oof,
            candidate_oof_path=args.oof_path,
            analysis_oof_path=args.analysis_oof_path,
        )
        Path(args.metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
