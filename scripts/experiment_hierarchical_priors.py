#!/usr/bin/env python3
"""Run CV experiments with fold-safe hierarchical target priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.config import CatBoostHyperParams, ID_COLUMN
from churn_baseline.data import infer_categorical_columns, load_csv, prepare_train_features
from churn_baseline.evaluation import binary_auc
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.modeling import (
    best_iteration_or_default,
    fit_full_train,
    fit_with_validation,
    predict_proba,
    save_model,
)
from churn_baseline.target_priors import (
    HierarchicalTargetPriorEncoder,
    build_default_prior_specs,
    build_default_numeric_deviation_columns,
    save_target_prior_encoder,
)


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
    raise ValueError(f"Could not find baseline AUC in {metrics_path}")


def _combine_features(
    base_features: pd.DataFrame,
    prior_features: pd.DataFrame,
    *,
    feature_mode: str,
) -> pd.DataFrame:
    if feature_mode == "priors_only":
        return prior_features.copy()
    if feature_mode == "raw_plus_priors":
        return pd.concat(
            [base_features.reset_index(drop=True), prior_features.reset_index(drop=True)],
            axis=1,
        )
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CV experiment with hierarchical target priors")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--base-feature-blocks",
        default="none",
        help="Optional base stateless blocks before priors (A,B,C,H,R,S,V,O,P or none).",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("raw_plus_priors", "priors_only"),
        default="raw_plus_priors",
        help="Use raw features plus priors, or only prior features.",
    )
    parser.add_argument(
        "--include-deviation-features",
        action="store_true",
        help="Add fold-safe cohort means and customer-vs-cohort deviations using the same hierarchy.",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/models/hierarchical_priors_cv.cbm",
        help="Output model path",
    )
    parser.add_argument(
        "--prior-encoder-path",
        default="artifacts/models/hierarchical_priors_encoder.json",
        help="Output path for fitted prior encoder metadata",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/hierarchical_priors_cv_metrics.json",
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/hierarchical_priors_cv_oof.csv",
        help="Output OOF predictions CSV path",
    )
    parser.add_argument(
        "--baseline-metrics-path",
        default="artifacts/reports/train_cv_multiseed_gate_s5_hiiter_metrics.json",
        help="JSON path with incumbent metric (ensemble_oof_auc/oof_auc/holdout_auc).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of Stratified K folds")
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
    if args.folds < 2:
        raise ValueError("folds must be >= 2")

    feature_blocks = _parse_feature_blocks(args.base_feature_blocks)
    params = CatBoostHyperParams(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
    )

    train_df = load_csv(args.train_csv)
    x_base, y = prepare_train_features(train_df, drop_id=True, feature_blocks=feature_blocks)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    oof_pred = pd.Series(index=x_base.index, dtype="float64", name="oof_pred")
    fold_rows: list[dict[str, float | int | None]] = []
    fold_iterations: list[int] = []
    prior_feature_columns: list[str] | None = None

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x_base, y), start=1):
        x_train_base = x_base.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        x_valid_base = x_base.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)

        encoder = HierarchicalTargetPriorEncoder(
            specs=build_default_prior_specs(),
            include_numeric_deviation=args.include_deviation_features,
        )
        x_train_prior = encoder.fit_transform(x_train_base, y_train)
        x_valid_prior = encoder.transform(x_valid_base)
        prior_feature_columns = list(x_train_prior.columns)

        x_train_model = _combine_features(
            x_train_base,
            x_train_prior,
            feature_mode=args.feature_mode,
        )
        x_valid_model = _combine_features(
            x_valid_base,
            x_valid_prior,
            feature_mode=args.feature_mode,
        )
        cat_columns = infer_categorical_columns(x_train_model)

        fold_params = CatBoostHyperParams(
            iterations=params.iterations,
            learning_rate=params.learning_rate,
            depth=params.depth,
            l2_leaf_reg=params.l2_leaf_reg,
            random_seed=args.random_state,
            loss_function=params.loss_function,
            eval_metric=params.eval_metric,
        )
        fold_model = fit_with_validation(
            x_train=x_train_model,
            y_train=y_train,
            x_valid=x_valid_model,
            y_valid=y_valid,
            cat_columns=cat_columns,
            params=fold_params,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=args.verbose,
        )
        fold_pred = predict_proba(fold_model, x_valid_model)
        oof_pred.iloc[valid_idx] = fold_pred.values

        fold_auc = binary_auc(y_valid, fold_pred)
        fold_final_iterations = best_iteration_or_default(fold_model, params.iterations)
        fold_iterations.append(fold_final_iterations)
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "auc": float(fold_auc),
                "best_iteration": int(fold_model.get_best_iteration()),
                "final_iterations": int(fold_final_iterations),
                "seed": int(args.random_state),
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError("OOF predictions contain missing values.")

    final_iterations = max(int(round(float(np.mean(fold_iterations)))), 1)
    full_encoder = HierarchicalTargetPriorEncoder(
        specs=build_default_prior_specs(),
        include_numeric_deviation=args.include_deviation_features,
    )
    x_full_prior = full_encoder.fit_transform(x_base.reset_index(drop=True), y.reset_index(drop=True))
    x_full_model = _combine_features(
        x_base.reset_index(drop=True),
        x_full_prior,
        feature_mode=args.feature_mode,
    )
    full_cat_columns = infer_categorical_columns(x_full_model)
    full_params = CatBoostHyperParams(
        iterations=final_iterations,
        learning_rate=params.learning_rate,
        depth=params.depth,
        l2_leaf_reg=params.l2_leaf_reg,
        random_seed=args.random_state,
        loss_function=params.loss_function,
        eval_metric=params.eval_metric,
    )
    full_model = fit_full_train(
        x_train=x_full_model,
        y_train=y.reset_index(drop=True),
        cat_columns=full_cat_columns,
        params=full_params,
        verbose=args.verbose,
    )
    save_model(full_model, args.model_path)
    save_target_prior_encoder(full_encoder, args.prior_encoder_path)

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "oof_pred": oof_pred.values,
        }
    )
    oof_out_path = Path(args.oof_path)
    oof_out_path.parent.mkdir(parents=True, exist_ok=True)
    oof_output.to_csv(oof_out_path, index=False)

    baseline_auc, baseline_key = _load_baseline_auc(args.baseline_metrics_path)
    cv_fold_aucs = [row["auc"] for row in fold_rows]
    metrics = {
        "train_rows": int(len(train_df)),
        "base_feature_count": int(x_base.shape[1]),
        "prior_feature_count": int(0 if prior_feature_columns is None else len(prior_feature_columns)),
        "feature_count": int(x_full_model.shape[1]),
        "base_feature_blocks": feature_blocks,
        "feature_mode": args.feature_mode,
        "include_deviation_features": bool(args.include_deviation_features),
        "deviation_numeric_columns": (
            list(build_default_numeric_deviation_columns()) if args.include_deviation_features else []
        ),
        "prior_specs": [spec.to_dict() for spec in build_default_prior_specs()],
        "categorical_columns": full_cat_columns,
        "cv_folds": int(args.folds),
        "cv_fold_metrics": fold_rows,
        "cv_mean_auc": float(np.mean(cv_fold_aucs)),
        "cv_std_auc": float(np.std(cv_fold_aucs)),
        "oof_auc": float(binary_auc(y, oof_pred)),
        "fold_final_iterations": [int(v) for v in fold_iterations],
        "final_iterations": int(final_iterations),
        "params_cv": params.to_catboost_kwargs(),
        "params_full_train": full_params.to_catboost_kwargs(),
        "model_path": str(args.model_path),
        "prior_encoder_path": str(args.prior_encoder_path),
        "oof_path": str(oof_out_path),
        "baseline_auc": float(baseline_auc),
        "baseline_auc_metric": baseline_key,
        "baseline_metrics_path": str(args.baseline_metrics_path),
        "delta_vs_baseline_auc": float(binary_auc(y, oof_pred) - baseline_auc),
        "target_column": "Churn",
        "id_column": ID_COLUMN,
    }
    write_json(args.metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
