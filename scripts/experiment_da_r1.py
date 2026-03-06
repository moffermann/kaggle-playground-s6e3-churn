#!/usr/bin/env python3
"""Run DA representation experiments (R1/R2) with fold-safe CV."""

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
from churn_baseline.data import (
    infer_categorical_columns,
    load_csv,
    prepare_train_features,
)
from churn_baseline.evaluation import binary_auc
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.modeling import (
    best_iteration_or_default,
    fit_full_train,
    fit_with_validation,
    predict_proba,
    save_model,
)
from churn_baseline.segment_features import (
    DEFAULT_OVERSAMPLE_POLICY,
    DEFAULT_OVERSAMPLE_SE_TARGET,
    DEFAULT_OVERSAMPLE_SMOOTHING_ALPHA,
    DEFAULT_RARE_SEGMENT_THRESHOLD,
    OVERSAMPLE_POLICY_VARIANCE_TARGET,
    SEGMENT_COMPONENT_COLUMNS,
    SEGMENT_COUNT_COLUMN,
    SEGMENT_FREQ_COLUMN,
    SegmentRepresentationEncoder,
    build_segment_sample_weights,
    oversample_rare_segments,
    save_segment_encoder,
)

DEFAULT_MODEL_PATH_R1 = "artifacts/models/da_r1_cv.cbm"
DEFAULT_SEGMENT_ENCODER_PATH_R1 = "artifacts/models/da_r1_segment_encoder.json"
DEFAULT_METRICS_PATH_R1 = "artifacts/reports/da_r1_cv_metrics.json"
DEFAULT_OOF_PATH_R1 = "artifacts/reports/da_r1_cv_oof.csv"

DEFAULT_MODEL_PATH_R2 = "artifacts/models/da_r2_cv.cbm"
DEFAULT_SEGMENT_ENCODER_PATH_R2 = "artifacts/models/da_r2_segment_encoder.json"
DEFAULT_METRICS_PATH_R2 = "artifacts/reports/da_r2_cv_metrics.json"
DEFAULT_OOF_PATH_R2 = "artifacts/reports/da_r2_cv_oof.csv"

DEFAULT_MODEL_PATH_R1W = "artifacts/models/da_r1w_cv.cbm"
DEFAULT_SEGMENT_ENCODER_PATH_R1W = "artifacts/models/da_r1w_segment_encoder.json"
DEFAULT_METRICS_PATH_R1W = "artifacts/reports/da_r1w_cv_metrics.json"
DEFAULT_OOF_PATH_R1W = "artifacts/reports/da_r1w_cv_oof.csv"

DEFAULT_MODEL_PATH_R2W = "artifacts/models/da_r2w_cv.cbm"
DEFAULT_SEGMENT_ENCODER_PATH_R2W = "artifacts/models/da_r2w_segment_encoder.json"
DEFAULT_METRICS_PATH_R2W = "artifacts/reports/da_r2w_cv_metrics.json"
DEFAULT_OOF_PATH_R2W = "artifacts/reports/da_r2w_cv_oof.csv"

DEFAULT_MODEL_PATH_R1O = "artifacts/models/da_r1o_cv.cbm"
DEFAULT_SEGMENT_ENCODER_PATH_R1O = "artifacts/models/da_r1o_segment_encoder.json"
DEFAULT_METRICS_PATH_R1O = "artifacts/reports/da_r1o_cv_metrics.json"
DEFAULT_OOF_PATH_R1O = "artifacts/reports/da_r1o_cv_oof.csv"

DEFAULT_MODEL_PATH_R2O = "artifacts/models/da_r2o_cv.cbm"
DEFAULT_SEGMENT_ENCODER_PATH_R2O = "artifacts/models/da_r2o_segment_encoder.json"
DEFAULT_METRICS_PATH_R2O = "artifacts/reports/da_r2o_cv_metrics.json"
DEFAULT_OOF_PATH_R2O = "artifacts/reports/da_r2o_cv_oof.csv"

DEFAULT_MODEL_PATH_R1OV = "artifacts/models/da_r1ov_cv.cbm"
DEFAULT_SEGMENT_ENCODER_PATH_R1OV = "artifacts/models/da_r1ov_segment_encoder.json"
DEFAULT_METRICS_PATH_R1OV = "artifacts/reports/da_r1ov_cv_metrics.json"
DEFAULT_OOF_PATH_R1OV = "artifacts/reports/da_r1ov_cv_oof.csv"

DEFAULT_MODEL_PATH_R2OV = "artifacts/models/da_r2ov_cv.cbm"
DEFAULT_SEGMENT_ENCODER_PATH_R2OV = "artifacts/models/da_r2ov_segment_encoder.json"
DEFAULT_METRICS_PATH_R2OV = "artifacts/reports/da_r2ov_cv_metrics.json"
DEFAULT_OOF_PATH_R2OV = "artifacts/reports/da_r2ov_cv_oof.csv"


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


def _normalize_default_paths(args: argparse.Namespace) -> None:
    """Switch default output paths by representation mode and DA toggle."""
    if args.representation_mode == "r2":
        if args.model_path == DEFAULT_MODEL_PATH_R1:
            args.model_path = DEFAULT_MODEL_PATH_R2
        if args.segment_encoder_path == DEFAULT_SEGMENT_ENCODER_PATH_R1:
            args.segment_encoder_path = DEFAULT_SEGMENT_ENCODER_PATH_R2
        if args.metrics_path == DEFAULT_METRICS_PATH_R1:
            args.metrics_path = DEFAULT_METRICS_PATH_R2
        if args.oof_path == DEFAULT_OOF_PATH_R1:
            args.oof_path = DEFAULT_OOF_PATH_R2

    if args.enable_segment_oversampling:
        if args.representation_mode == "r1":
            if args.oversample_policy == OVERSAMPLE_POLICY_VARIANCE_TARGET:
                if args.model_path in {DEFAULT_MODEL_PATH_R1, DEFAULT_MODEL_PATH_R1O}:
                    args.model_path = DEFAULT_MODEL_PATH_R1OV
                if args.segment_encoder_path in {
                    DEFAULT_SEGMENT_ENCODER_PATH_R1,
                    DEFAULT_SEGMENT_ENCODER_PATH_R1O,
                }:
                    args.segment_encoder_path = DEFAULT_SEGMENT_ENCODER_PATH_R1OV
                if args.metrics_path in {DEFAULT_METRICS_PATH_R1, DEFAULT_METRICS_PATH_R1O}:
                    args.metrics_path = DEFAULT_METRICS_PATH_R1OV
                if args.oof_path in {DEFAULT_OOF_PATH_R1, DEFAULT_OOF_PATH_R1O}:
                    args.oof_path = DEFAULT_OOF_PATH_R1OV
            else:
                if args.model_path == DEFAULT_MODEL_PATH_R1:
                    args.model_path = DEFAULT_MODEL_PATH_R1O
                if args.segment_encoder_path == DEFAULT_SEGMENT_ENCODER_PATH_R1:
                    args.segment_encoder_path = DEFAULT_SEGMENT_ENCODER_PATH_R1O
                if args.metrics_path == DEFAULT_METRICS_PATH_R1:
                    args.metrics_path = DEFAULT_METRICS_PATH_R1O
                if args.oof_path == DEFAULT_OOF_PATH_R1:
                    args.oof_path = DEFAULT_OOF_PATH_R1O
            return

        if args.oversample_policy == OVERSAMPLE_POLICY_VARIANCE_TARGET:
            if args.model_path in {DEFAULT_MODEL_PATH_R2, DEFAULT_MODEL_PATH_R2O}:
                args.model_path = DEFAULT_MODEL_PATH_R2OV
            if args.segment_encoder_path in {
                DEFAULT_SEGMENT_ENCODER_PATH_R2,
                DEFAULT_SEGMENT_ENCODER_PATH_R2O,
            }:
                args.segment_encoder_path = DEFAULT_SEGMENT_ENCODER_PATH_R2OV
            if args.metrics_path in {DEFAULT_METRICS_PATH_R2, DEFAULT_METRICS_PATH_R2O}:
                args.metrics_path = DEFAULT_METRICS_PATH_R2OV
            if args.oof_path in {DEFAULT_OOF_PATH_R2, DEFAULT_OOF_PATH_R2O}:
                args.oof_path = DEFAULT_OOF_PATH_R2OV
        else:
            if args.model_path == DEFAULT_MODEL_PATH_R2:
                args.model_path = DEFAULT_MODEL_PATH_R2O
            if args.segment_encoder_path == DEFAULT_SEGMENT_ENCODER_PATH_R2:
                args.segment_encoder_path = DEFAULT_SEGMENT_ENCODER_PATH_R2O
            if args.metrics_path == DEFAULT_METRICS_PATH_R2:
                args.metrics_path = DEFAULT_METRICS_PATH_R2O
            if args.oof_path == DEFAULT_OOF_PATH_R2:
                args.oof_path = DEFAULT_OOF_PATH_R2O
        return

    if not args.enable_segment_reweighting:
        return

    if args.representation_mode == "r1":
        if args.model_path == DEFAULT_MODEL_PATH_R1:
            args.model_path = DEFAULT_MODEL_PATH_R1W
        if args.segment_encoder_path == DEFAULT_SEGMENT_ENCODER_PATH_R1:
            args.segment_encoder_path = DEFAULT_SEGMENT_ENCODER_PATH_R1W
        if args.metrics_path == DEFAULT_METRICS_PATH_R1:
            args.metrics_path = DEFAULT_METRICS_PATH_R1W
        if args.oof_path == DEFAULT_OOF_PATH_R1:
            args.oof_path = DEFAULT_OOF_PATH_R1W
        return

    if args.model_path == DEFAULT_MODEL_PATH_R2:
        args.model_path = DEFAULT_MODEL_PATH_R2W
    if args.segment_encoder_path == DEFAULT_SEGMENT_ENCODER_PATH_R2:
        args.segment_encoder_path = DEFAULT_SEGMENT_ENCODER_PATH_R2W
    if args.metrics_path == DEFAULT_METRICS_PATH_R2:
        args.metrics_path = DEFAULT_METRICS_PATH_R2W
    if args.oof_path == DEFAULT_OOF_PATH_R2:
        args.oof_path = DEFAULT_OOF_PATH_R2W


def _summarize_weights(weights: pd.Series) -> dict[str, float | int]:
    values = weights.astype("float64")
    return {
        "weight_mean": float(values.mean()),
        "weight_std": float(values.std(ddof=0)),
        "weight_min": float(values.min()),
        "weight_max": float(values.max()),
        "weight_p95": float(values.quantile(0.95)),
        "upweighted_rows": int((values > 1.0).sum()),
        "upweighted_rate": float((values > 1.0).mean()),
    }


def _parse_se_grid(raw: str) -> list[float]:
    text = raw.strip()
    if not text:
        return []
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DA representation experiment (R1/R2) with fold-safe CV"
    )
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--representation-mode",
        choices=["r1", "r2"],
        default="r1",
        help="R1: segment key/count/freq/rare. R2: R1 + rare bucketing + component freq.",
    )
    parser.add_argument(
        "--base-feature-blocks",
        default="none",
        help="Optional stateless blocks before representation features (A,B,C,O,P or none).",
    )
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=DEFAULT_RARE_SEGMENT_THRESHOLD,
        help="Rare segment threshold over segment_freq (default 0.005 == 0.5%%).",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH_R1,
        help="Output model path",
    )
    parser.add_argument(
        "--segment-encoder-path",
        default=DEFAULT_SEGMENT_ENCODER_PATH_R1,
        help="Output path for fitted segment encoder metadata",
    )
    parser.add_argument(
        "--metrics-path",
        default=DEFAULT_METRICS_PATH_R1,
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--oof-path",
        default=DEFAULT_OOF_PATH_R1,
        help="Output OOF predictions CSV path",
    )
    parser.add_argument(
        "--baseline-metrics-path",
        default="artifacts/reports/train_cv_multiseed_full_hiiter_metrics.json",
        help="JSON path with incumbent metric (ensemble_oof_auc/oof_auc/holdout_auc)",
    )
    parser.add_argument(
        "--enable-segment-reweighting",
        action="store_true",
        help="Apply fold-safe sample weighting from segment rarity.",
    )
    parser.add_argument(
        "--enable-segment-oversampling",
        action="store_true",
        help="Apply conservative in-fold oversampling for rare segments.",
    )
    parser.add_argument(
        "--oversample-policy",
        choices=["fixed", "variance_target"],
        default=DEFAULT_OVERSAMPLE_POLICY,
        help="Oversampling policy: fixed frequency target or variance-targeted allocation.",
    )
    parser.add_argument(
        "--segment-weight-power",
        type=float,
        default=0.5,
        help="Exponent for rarity ratio when building sample_weight.",
    )
    parser.add_argument(
        "--segment-weight-max",
        type=float,
        default=3.0,
        help="Max cap for per-row sample_weight.",
    )
    parser.add_argument(
        "--segment-weight-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize sample_weight to mean 1.0 (recommended).",
    )
    parser.add_argument(
        "--oversample-target-freq",
        type=float,
        default=DEFAULT_RARE_SEGMENT_THRESHOLD,
        help="Target minimum segment frequency after conservative oversampling.",
    )
    parser.add_argument(
        "--oversample-max-multiplier",
        type=float,
        default=2.0,
        help="Per-segment cap for oversampling as a multiplier over original count.",
    )
    parser.add_argument(
        "--oversample-max-added-rate",
        type=float,
        default=0.03,
        help="Global cap for added rows (fraction of train rows per fold).",
    )
    parser.add_argument(
        "--oversample-random-state",
        type=int,
        default=42,
        help="Base random seed for oversampling.",
    )
    parser.add_argument(
        "--oversample-se-target",
        type=float,
        default=DEFAULT_OVERSAMPLE_SE_TARGET,
        help="Target standard error for segment churn-rate estimate (variance_target policy).",
    )
    parser.add_argument(
        "--oversample-smoothing-alpha",
        type=float,
        default=DEFAULT_OVERSAMPLE_SMOOTHING_ALPHA,
        help="Additive smoothing weight for segment churn-rate estimate (variance_target policy).",
    )
    parser.add_argument(
        "--oversample-plan-only",
        action="store_true",
        help="Only compute oversampling plan summary (no model training).",
    )
    parser.add_argument(
        "--oversample-plan-se-grid",
        default="",
        help="Comma-separated SE targets for plan-only mode (variance_target policy).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of Stratified K folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--iterations", type=int, default=12000, help="Max boosting rounds")
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
    if args.enable_segment_reweighting and args.enable_segment_oversampling:
        raise ValueError("Use either segment reweighting or segment oversampling, not both.")
    if args.oversample_plan_only and not args.enable_segment_oversampling:
        raise ValueError("oversample-plan-only requires --enable-segment-oversampling.")
    _normalize_default_paths(args)
    if args.folds < 2:
        raise ValueError("folds must be >= 2")
    if args.rare_threshold <= 0 or args.rare_threshold >= 1:
        raise ValueError("rare-threshold must be in (0, 1)")
    if args.segment_weight_power <= 0:
        raise ValueError("segment-weight-power must be > 0")
    if args.segment_weight_max < 1:
        raise ValueError("segment-weight-max must be >= 1")
    if args.oversample_target_freq <= 0 or args.oversample_target_freq >= 1:
        raise ValueError("oversample-target-freq must be in (0, 1)")
    if args.oversample_max_multiplier < 1:
        raise ValueError("oversample-max-multiplier must be >= 1")
    if args.oversample_max_added_rate < 0 or args.oversample_max_added_rate >= 1:
        raise ValueError("oversample-max-added-rate must be in [0, 1)")
    if args.oversample_se_target <= 0:
        raise ValueError("oversample-se-target must be > 0")
    if args.oversample_smoothing_alpha < 0:
        raise ValueError("oversample-smoothing-alpha must be >= 0")
    se_grid = _parse_se_grid(args.oversample_plan_se_grid)
    if se_grid and args.oversample_policy != OVERSAMPLE_POLICY_VARIANCE_TARGET:
        raise ValueError("oversample-plan-se-grid requires --oversample-policy variance_target.")
    if any(value <= 0 for value in se_grid):
        raise ValueError("oversample-plan-se-grid values must be > 0.")

    base_feature_blocks = _parse_feature_blocks(args.base_feature_blocks)
    params = CatBoostHyperParams(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
    )

    train_df = load_csv(args.train_csv)
    x_base, y = prepare_train_features(train_df, drop_id=True, feature_blocks=base_feature_blocks)

    if args.oversample_plan_only:
        plan_encoder = SegmentRepresentationEncoder(
            component_columns=SEGMENT_COMPONENT_COLUMNS,
            rare_segment_threshold=args.rare_threshold,
            enable_rare_bucketing=args.representation_mode == "r2",
        )
        x_plan = plan_encoder.fit_transform(x_base)
        se_targets = se_grid or [float(args.oversample_se_target)]
        plan_rows = []
        for se_target in se_targets:
            _, _, summary = oversample_rare_segments(
                x_plan,
                y,
                target_min_freq=args.oversample_target_freq,
                max_multiplier=args.oversample_max_multiplier,
                max_added_rate=args.oversample_max_added_rate,
                policy=args.oversample_policy,
                se_target=float(se_target),
                smoothing_alpha=args.oversample_smoothing_alpha,
                random_state=args.oversample_random_state,
            )
            plan_rows.append(
                {
                    "se_target": float(se_target),
                    "summary": summary,
                }
            )

        plan_report = {
            "plan_only": True,
            "train_rows": int(len(train_df)),
            "representation_mode": args.representation_mode,
            "oversample_policy": str(args.oversample_policy),
            "oversample_target_freq": float(args.oversample_target_freq),
            "oversample_max_multiplier": float(args.oversample_max_multiplier),
            "oversample_max_added_rate": float(args.oversample_max_added_rate),
            "oversample_smoothing_alpha": float(args.oversample_smoothing_alpha),
            "rare_threshold": float(args.rare_threshold),
            "se_targets": [float(v) for v in se_targets],
            "plan": plan_rows,
        }
        print(json.dumps(plan_report, indent=2))
        return 0

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    oof_pred = pd.Series(index=x_base.index, dtype="float64", name=f"oof_{args.representation_mode}")

    fold_rows = []
    fold_final_iterations = []
    fold_unseen_counts = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x_base, y), start=1):
        x_train_fold = x_base.iloc[train_idx].copy()
        x_valid_fold = x_base.iloc[valid_idx].copy()
        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]

        encoder = SegmentRepresentationEncoder(
            component_columns=SEGMENT_COMPONENT_COLUMNS,
            rare_segment_threshold=args.rare_threshold,
            enable_rare_bucketing=args.representation_mode == "r2",
        )
        x_train_fold = encoder.fit_transform(x_train_fold)
        x_valid_fold = encoder.transform(x_valid_fold)
        unseen_in_valid = int((x_valid_fold[SEGMENT_COUNT_COLUMN] == 0).sum())
        fold_unseen_counts.append(unseen_in_valid)
        fold_oversampling_summary: dict[str, float | int] | None = None
        if args.enable_segment_oversampling:
            x_train_fold, y_train_fold, fold_oversampling_summary = oversample_rare_segments(
                x_train_fold,
                y_train_fold,
                target_min_freq=args.oversample_target_freq,
                max_multiplier=args.oversample_max_multiplier,
                max_added_rate=args.oversample_max_added_rate,
                policy=args.oversample_policy,
                se_target=args.oversample_se_target,
                smoothing_alpha=args.oversample_smoothing_alpha,
                random_state=args.oversample_random_state + fold_number,
            )

        fold_weight_summary: dict[str, float | int] | None = None
        sample_weight_train: pd.Series | None = None
        if args.enable_segment_reweighting:
            sample_weight_train = build_segment_sample_weights(
                x_train_fold[SEGMENT_FREQ_COLUMN],
                rare_threshold=args.rare_threshold,
                power=args.segment_weight_power,
                max_weight=args.segment_weight_max,
                normalize=args.segment_weight_normalize,
            )
            fold_weight_summary = _summarize_weights(sample_weight_train)

        cat_columns = infer_categorical_columns(x_train_fold)
        fold_model = fit_with_validation(
            x_train=x_train_fold,
            y_train=y_train_fold,
            x_valid=x_valid_fold,
            y_valid=y_valid_fold,
            cat_columns=cat_columns,
            params=params,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=args.verbose,
            sample_weight=sample_weight_train,
        )

        fold_pred = predict_proba(fold_model, x_valid_fold)
        oof_pred.iloc[valid_idx] = fold_pred.values
        fold_auc = binary_auc(y_valid_fold, fold_pred)
        final_iterations = best_iteration_or_default(fold_model, params.iterations)
        fold_final_iterations.append(final_iterations)
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "auc": float(fold_auc),
                "best_iteration": int(fold_model.get_best_iteration()),
                "final_iterations": int(final_iterations),
                "unseen_segments_in_valid": unseen_in_valid,
                "seed": int(args.random_state),
                "segment_reweighting_enabled": bool(args.enable_segment_reweighting),
                "segment_oversampling_enabled": bool(args.enable_segment_oversampling),
                "segment_oversampling_summary": fold_oversampling_summary,
                "segment_weight_summary": fold_weight_summary,
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError("OOF predictions contain missing values")

    cv_fold_aucs = [row["auc"] for row in fold_rows]
    final_iterations = max(int(round(float(np.mean(fold_final_iterations)))), 1)

    full_encoder = SegmentRepresentationEncoder(
        component_columns=SEGMENT_COMPONENT_COLUMNS,
        rare_segment_threshold=args.rare_threshold,
        enable_rare_bucketing=args.representation_mode == "r2",
    )
    x_full = full_encoder.fit_transform(x_base)
    full_cat_columns = infer_categorical_columns(x_full)
    full_train_oversampling_summary: dict[str, float | int] | None = None
    x_full_train = x_full
    y_full_train = y
    if args.enable_segment_oversampling:
        x_full_train, y_full_train, full_train_oversampling_summary = oversample_rare_segments(
            x_full,
            y,
            target_min_freq=args.oversample_target_freq,
            max_multiplier=args.oversample_max_multiplier,
            max_added_rate=args.oversample_max_added_rate,
            policy=args.oversample_policy,
            se_target=args.oversample_se_target,
            smoothing_alpha=args.oversample_smoothing_alpha,
            random_state=args.oversample_random_state,
        )

    full_train_weight_summary: dict[str, float | int] | None = None
    full_train_sample_weight: pd.Series | None = None
    if args.enable_segment_reweighting:
        full_train_sample_weight = build_segment_sample_weights(
            x_full_train[SEGMENT_FREQ_COLUMN],
            rare_threshold=args.rare_threshold,
            power=args.segment_weight_power,
            max_weight=args.segment_weight_max,
            normalize=args.segment_weight_normalize,
        )
        full_train_weight_summary = _summarize_weights(full_train_sample_weight)

    full_params = CatBoostHyperParams(
        iterations=final_iterations,
        learning_rate=params.learning_rate,
        depth=params.depth,
        l2_leaf_reg=params.l2_leaf_reg,
        random_seed=params.random_seed,
        loss_function=params.loss_function,
        eval_metric=params.eval_metric,
    )
    full_model = fit_full_train(
        x_train=x_full_train,
        y_train=y_full_train,
        cat_columns=full_cat_columns,
        params=full_params,
        verbose=args.verbose,
        sample_weight=full_train_sample_weight,
    )

    save_model(full_model, args.model_path)
    save_segment_encoder(full_encoder, args.segment_encoder_path)

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN],
            "target": y.astype(int).values,
            "oof_pred": oof_pred.values,
        }
    )
    oof_out_path = Path(args.oof_path)
    oof_out_path.parent.mkdir(parents=True, exist_ok=True)
    oof_output.to_csv(oof_out_path, index=False)

    baseline_auc, baseline_key = _load_baseline_auc(args.baseline_metrics_path)
    oof_auc = binary_auc(y, oof_pred)

    metrics = {
        "train_rows": int(len(train_df)),
        "full_train_rows_after_augmentation": int(len(x_full_train)),
        "feature_count": int(x_full.shape[1]),
        "base_feature_blocks": base_feature_blocks,
        "representation_mode": args.representation_mode,
        "segment_reweighting_enabled": bool(args.enable_segment_reweighting),
        "segment_oversampling_enabled": bool(args.enable_segment_oversampling),
        "segment_weight_power": float(args.segment_weight_power),
        "segment_weight_max": float(args.segment_weight_max),
        "segment_weight_normalize": bool(args.segment_weight_normalize),
        "oversample_policy": str(args.oversample_policy),
        "oversample_target_freq": float(args.oversample_target_freq),
        "oversample_max_multiplier": float(args.oversample_max_multiplier),
        "oversample_max_added_rate": float(args.oversample_max_added_rate),
        "oversample_random_state": int(args.oversample_random_state),
        "oversample_se_target": float(args.oversample_se_target),
        "oversample_smoothing_alpha": float(args.oversample_smoothing_alpha),
        "da_r1_enabled": args.representation_mode == "r1",
        "da_r2_enabled": args.representation_mode == "r2",
        "segment_component_columns": list(SEGMENT_COMPONENT_COLUMNS),
        "rare_threshold": float(args.rare_threshold),
        "categorical_columns": full_cat_columns,
        "cv_folds": int(args.folds),
        "cv_fold_metrics": fold_rows,
        "cv_mean_auc": float(np.mean(cv_fold_aucs)),
        "cv_std_auc": float(np.std(cv_fold_aucs)),
        "oof_auc": float(oof_auc),
        "fold_final_iterations": [int(v) for v in fold_final_iterations],
        "final_iterations": int(final_iterations),
        "unseen_segments_valid_total": int(sum(fold_unseen_counts)),
        "unseen_segments_valid_mean": float(np.mean(fold_unseen_counts)),
        "full_train_oversampling_summary": full_train_oversampling_summary,
        "full_train_weight_summary": full_train_weight_summary,
        "params_cv": params.to_catboost_kwargs(),
        "params_full_train": full_params.to_catboost_kwargs(),
        "model_path": str(args.model_path),
        "segment_encoder_path": str(args.segment_encoder_path),
        "oof_path": str(oof_out_path),
        "baseline_auc": float(baseline_auc),
        "baseline_auc_metric": baseline_key,
        "baseline_metrics_path": str(args.baseline_metrics_path),
        "delta_vs_baseline_auc": float(oof_auc - baseline_auc),
        "experiment_name": (
            f"da_{args.representation_mode}_segment_representation"
            + (
                f"_oversampled_{args.oversample_policy}"
                if args.enable_segment_oversampling
                else ""
            )
            + ("_reweighted" if args.enable_segment_reweighting else "")
        ),
        "id_column": ID_COLUMN,
    }
    write_json(args.metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
