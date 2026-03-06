#!/usr/bin/env python3
"""Train LightGBM with Stratified K-Fold CV and save OOF artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.config import ID_COLUMN
from churn_baseline.data import load_csv, prepare_train_features
from churn_baseline.encoded_features import build_dense_encoded_matrix
from churn_baseline.evaluation import binary_auc
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.oof_tools import load_baseline_auc


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM CV baseline")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--feature-blocks",
        default="none",
        help="Optional feature blocks (A,B,C,O,P) or none.",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/models/lightgbm_cv.txt",
        help="Output full-train model path",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/train_lightgbm_cv_metrics.json",
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/train_lightgbm_cv_oof.csv",
        help="Output OOF predictions CSV path",
    )
    parser.add_argument(
        "--baseline-metrics-path",
        default="artifacts/reports/train_cv_multiseed_full_hiiter_metrics.json",
        help="JSON path with incumbent metric (ensemble_oof_auc/oof_auc/holdout_auc).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of Stratified K folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--iterations", type=int, default=12000, help="Max boosting rounds")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--num-leaves", type=int, default=63, help="Number of leaves")
    parser.add_argument("--max-depth", type=int, default=-1, help="Tree max depth")
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=50,
        help="Minimum data in one leaf",
    )
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsample ratio")
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Feature subsample ratio",
    )
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularization")
    parser.add_argument("--reg-lambda", type=float, default=0.0, help="L2 regularization")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=200,
        help="Early stopping rounds",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel worker count")
    parser.add_argument(
        "--verbose-eval",
        type=int,
        default=200,
        help="LightGBM log evaluation interval (0 disables fold logs).",
    )
    return parser.parse_args()


def _build_lgbm_model(args: argparse.Namespace, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        max_depth=int(args.max_depth),
        min_child_samples=int(args.min_child_samples),
        subsample=float(args.subsample),
        colsample_bytree=float(args.colsample_bytree),
        reg_alpha=float(args.reg_alpha),
        reg_lambda=float(args.reg_lambda),
        random_state=int(args.random_state),
        n_jobs=int(args.n_jobs),
    )


def _build_callbacks(args: argparse.Namespace) -> list:
    callbacks = [lgb.early_stopping(stopping_rounds=int(args.early_stopping_rounds), verbose=False)]
    if args.verbose_eval > 0:
        callbacks.append(lgb.log_evaluation(period=int(args.verbose_eval)))
    return callbacks


def main() -> int:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("folds must be >= 2")
    if args.iterations < 1:
        raise ValueError("iterations must be >= 1")

    feature_blocks = _parse_feature_blocks(args.feature_blocks)
    train_df = load_csv(args.train_csv)
    x_raw, y = prepare_train_features(train_df, drop_id=True, feature_blocks=feature_blocks)
    x = build_dense_encoded_matrix(x_raw)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    oof_pred = pd.Series(index=x.index, dtype="float64", name="oof_pred")

    fold_rows = []
    fold_final_iterations = []
    callbacks = _build_callbacks(args)

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = _build_lgbm_model(args, n_estimators=args.iterations)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=callbacks,
        )

        fold_pred = model.predict_proba(x_valid)[:, 1]
        oof_pred.iloc[valid_idx] = fold_pred
        fold_auc = binary_auc(y_valid, fold_pred)
        best_iter = int(model.best_iteration_) if model.best_iteration_ is not None else int(args.iterations)
        best_iter = max(best_iter, 1)
        fold_final_iterations.append(best_iter)
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "auc": float(fold_auc),
                "final_iterations": int(best_iter),
                "seed": int(args.random_state),
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError("OOF predictions contain missing values.")

    final_iterations = max(int(round(float(np.mean(fold_final_iterations)))), 1)
    full_model = _build_lgbm_model(args, n_estimators=final_iterations)
    full_model.fit(x, y)

    model_out_path = Path(args.model_path)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    full_model.booster_.save_model(str(model_out_path))

    oof_out = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "oof_pred": oof_pred.values,
        }
    )
    oof_out_path = Path(args.oof_path)
    oof_out_path.parent.mkdir(parents=True, exist_ok=True)
    oof_out.to_csv(oof_out_path, index=False)

    baseline_auc, baseline_key = load_baseline_auc(args.baseline_metrics_path)
    oof_auc = binary_auc(y, oof_pred)
    cv_fold_aucs = [row["auc"] for row in fold_rows]

    metrics = {
        "train_rows": int(len(train_df)),
        "feature_count": int(x.shape[1]),
        "feature_blocks": feature_blocks,
        "cv_folds": int(args.folds),
        "cv_fold_metrics": fold_rows,
        "cv_mean_auc": float(np.mean(cv_fold_aucs)),
        "cv_std_auc": float(np.std(cv_fold_aucs)),
        "oof_auc": float(oof_auc),
        "fold_final_iterations": [int(v) for v in fold_final_iterations],
        "final_iterations": int(final_iterations),
        "params": {
            "iterations": int(args.iterations),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "max_depth": int(args.max_depth),
            "min_child_samples": int(args.min_child_samples),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_alpha": float(args.reg_alpha),
            "reg_lambda": float(args.reg_lambda),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
            "early_stopping_rounds": int(args.early_stopping_rounds),
        },
        "model_family": "lightgbm",
        "model_path": str(model_out_path),
        "oof_path": str(oof_out_path),
        "baseline_auc": float(baseline_auc),
        "baseline_auc_metric": baseline_key,
        "baseline_metrics_path": str(args.baseline_metrics_path),
        "delta_vs_baseline_auc": float(oof_auc - baseline_auc),
        "id_column": ID_COLUMN,
        "target_column": "target",
    }
    write_json(args.metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
