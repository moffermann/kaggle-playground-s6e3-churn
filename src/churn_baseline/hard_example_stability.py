"""Example-stability scoring and local reranking against incumbent v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold

from .artifacts import ensure_parent_dir, write_json
from .config import CatBoostHyperParams, ID_COLUMN, TARGET_COLUMN
from .data import infer_categorical_columns, load_csv
from .diagnostics import build_family_frame
from .evaluation import binary_auc
from .modeling import best_iteration_or_default, fit_with_validation, save_model
from .pipeline import _prepare_train_matrix, _transform_pair_with_stateful_blocks, _transform_single_with_stateful_blocks
from .specialist import _append_reference_features, _clip_probability
from .uncertainty_band import _align_reference_prediction, _build_component_stats


MIN_MASKED_TRAIN_ROWS = 2000
MIN_MASKED_VALID_ROWS = 500


def _build_stability_oof(
    *,
    train_df: pd.DataFrame,
    feature_blocks: Sequence[str] | None,
    params: CatBoostHyperParams,
    folds: int,
    repeats: int,
    random_state: int,
    early_stopping_rounds: int,
    verbose: int,
) -> tuple[pd.DataFrame, pd.Series, tuple[str, ...], tuple[str, ...], pd.DataFrame, dict[str, Any]]:
    normalized_blocks, _, stateful_blocks, x, y = _prepare_train_matrix(train_df, feature_blocks)
    pred_matrix = np.full((len(x), int(repeats)), np.nan, dtype="float64")
    repeat_rows: list[dict[str, Any]] = []

    for repeat in range(int(repeats)):
        splitter = StratifiedKFold(
            n_splits=int(folds),
            shuffle=True,
            random_state=int(random_state) + repeat,
        )
        repeat_pred = pd.Series(index=x.index, dtype="float64", name=f"repeat_{repeat}_pred")
        fold_iterations: list[int] = []
        fold_aucs: list[float] = []

        for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
            x_train = x.iloc[train_idx]
            y_train = y.iloc[train_idx]
            x_valid = x.iloc[valid_idx]
            y_valid = y.iloc[valid_idx]
            x_train, x_valid = _transform_pair_with_stateful_blocks(x_train, x_valid, stateful_blocks)
            cat_columns = infer_categorical_columns(x_train)

            fold_params = CatBoostHyperParams(
                iterations=params.iterations,
                learning_rate=params.learning_rate,
                depth=params.depth,
                l2_leaf_reg=params.l2_leaf_reg,
                random_seed=int(random_state) + repeat * 100 + fold_number,
            )
            model = fit_with_validation(
                x_train,
                y_train,
                x_valid,
                y_valid,
                cat_columns,
                fold_params,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
            )
            valid_pred = pd.Series(
                model.predict_proba(x_valid)[:, 1],
                index=x_valid.index,
                dtype="float64",
            )
            repeat_pred.iloc[valid_idx] = valid_pred.values
            fold_iterations.append(best_iteration_or_default(model, fold_params.iterations))
            fold_aucs.append(float(binary_auc(y_valid, valid_pred)))

        if repeat_pred.isna().any():
            raise RuntimeError(f"stability OOF repeat {repeat} contains missing values")
        pred_matrix[:, repeat] = repeat_pred.to_numpy(dtype="float64")
        repeat_rows.append(
            {
                "repeat": int(repeat),
                "mean_auc": float(np.mean(fold_aucs)),
                "min_auc": float(np.min(fold_aucs)),
                "max_auc": float(np.max(fold_aucs)),
                "mean_final_iterations": float(np.mean(fold_iterations)),
            }
        )

    pred_mean = pred_matrix.mean(axis=1)
    pred_std = pred_matrix.std(axis=1)
    pred_min = pred_matrix.min(axis=1)
    pred_max = pred_matrix.max(axis=1)
    vote_share = (pred_matrix >= 0.5).mean(axis=1)

    stability_frame = pd.DataFrame(
        {
            "stability_pred_mean": pred_mean,
            "stability_pred_std": pred_std,
            "stability_pred_min": pred_min,
            "stability_pred_max": pred_max,
            "stability_pred_range": pred_max - pred_min,
            "stability_positive_vote_share": vote_share,
            "stability_flip_rate": np.minimum(vote_share, 1.0 - vote_share),
            "hard_example_score": pred_std + np.minimum(vote_share, 1.0 - vote_share),
        },
        index=x.index,
    )

    metadata = {
        "feature_blocks": list(normalized_blocks),
        "feature_count": int(x.shape[1]),
        "feature_columns": list(x.columns),
        "repeats": int(repeats),
        "folds": int(folds),
        "repeat_metrics": repeat_rows,
    }
    return x, y, tuple(normalized_blocks), tuple(stateful_blocks), stability_frame, metadata


def _build_hard_example_mask(
    *,
    train_df: pd.DataFrame,
    reference_pred: pd.Series,
    family_level: str,
    family_value: str,
    stability_frame: pd.DataFrame,
    hard_score_quantile: float,
    reference_band_half_width: float | None,
) -> tuple[pd.Series, dict[str, Any]]:
    if family_level not in {"segment3", "segment5"}:
        raise ValueError("family_level must be 'segment3' or 'segment5'")
    if float(hard_score_quantile) <= 0.0 or float(hard_score_quantile) >= 1.0:
        raise ValueError(f"hard_score_quantile must be in (0,1), got {hard_score_quantile}")
    if reference_band_half_width is not None and (float(reference_band_half_width) <= 0.0 or float(reference_band_half_width) >= 0.5):
        raise ValueError(f"reference_band_half_width must be in (0,0.5), got {reference_band_half_width}")

    family_frame = build_family_frame(train_df)
    family_mask = family_frame[family_level].astype(str).eq(str(family_value))
    family_rows = int(family_mask.sum())
    if family_rows <= 0:
        raise ValueError(f"family '{family_value}' not found in train.csv for level '{family_level}'")

    family_scores = stability_frame.loc[family_mask, "hard_example_score"].astype("float64")
    score_threshold = float(family_scores.quantile(float(hard_score_quantile)))
    mask = family_mask & stability_frame["hard_example_score"].ge(score_threshold)

    reference_uncertainty = reference_pred.sub(0.5).abs()
    band_rows = None
    if reference_band_half_width is not None:
        band_mask = reference_uncertainty.le(float(reference_band_half_width))
        band_rows = int((family_mask & band_mask).sum())
        mask = mask & band_mask

    mask_rows = int(mask.sum())
    mask_summary = {
        "family_level": str(family_level),
        "family_value": str(family_value),
        "family_rows": family_rows,
        "hard_score_quantile": float(hard_score_quantile),
        "score_threshold": score_threshold,
        "reference_band_half_width": None if reference_band_half_width is None else float(reference_band_half_width),
        "reference_band_rows_within_family": band_rows,
        "mask_rows": mask_rows,
        "mask_rate": float(mask_rows / max(len(train_df), 1)),
        "family_rate": float(family_rows / max(len(train_df), 1)),
        "hard_score_mean_on_mask": float(stability_frame.loc[mask, "hard_example_score"].mean()) if mask_rows else None,
        "hard_score_mean_in_family": float(family_scores.mean()),
        "stability_std_mean_on_mask": float(stability_frame.loc[mask, "stability_pred_std"].mean()) if mask_rows else None,
        "flip_rate_mean_on_mask": float(stability_frame.loc[mask, "stability_flip_rate"].mean()) if mask_rows else None,
        "reference_uncertainty_mean_on_mask": float(reference_uncertainty.loc[mask].mean()) if mask_rows else None,
    }
    return mask.astype(bool), mask_summary


def run_hard_example_stability_reranker_cv(
    *,
    train_csv_path: str | Path,
    reference_frame: pd.DataFrame,
    reference_component_frame: pd.DataFrame | None,
    stability_feature_blocks: Sequence[str] | None,
    stability_params: CatBoostHyperParams,
    stability_repeats: int,
    reranker_feature_blocks: Sequence[str] | None,
    reranker_params: CatBoostHyperParams,
    folds: int,
    random_state: int,
    early_stopping_rounds: int,
    verbose: int,
    alpha_grid: Sequence[float],
    family_level: str,
    family_value: str,
    hard_score_quantile: float,
    reference_band_half_width: float | None,
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one value")

    train_df = load_csv(train_csv_path)
    reference_pred = _align_reference_prediction(train_df, reference_frame)
    component_frame, _ = _build_component_stats(train_df[ID_COLUMN], reference_component_frame)

    _, _, stability_blocks_norm, _, stability_frame, stability_meta = _build_stability_oof(
        train_df=train_df,
        feature_blocks=stability_feature_blocks,
        params=stability_params,
        folds=folds,
        repeats=stability_repeats,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )

    reranker_blocks_norm, _, stateful_blocks, x, y = _prepare_train_matrix(train_df, reranker_feature_blocks)
    x = x.copy()
    x[stability_frame.columns] = stability_frame.astype("float64")
    x["stability_mean_minus_reference"] = stability_frame["stability_pred_mean"].astype("float64") - reference_pred.astype("float64")
    x["stability_mean_minus_reference_abs"] = x["stability_mean_minus_reference"].abs()

    hard_mask, mask_summary = _build_hard_example_mask(
        train_df=train_df,
        reference_pred=reference_pred,
        family_level=family_level,
        family_value=family_value,
        stability_frame=stability_frame,
        hard_score_quantile=hard_score_quantile,
        reference_band_half_width=reference_band_half_width,
    )

    masked_rows = int(hard_mask.sum())
    if masked_rows < MIN_MASKED_TRAIN_ROWS:
        raise ValueError(f"hard-example mask selects only {masked_rows} rows. Minimum required: {MIN_MASKED_TRAIN_ROWS}")
    if y.loc[hard_mask].nunique() < 2:
        raise ValueError("hard-example mask must contain both classes")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    residual_pred = pd.Series(index=x.index, dtype="float64", name="residual_pred")
    fold_rows: list[dict[str, Any]] = []
    fold_iterations: list[int] = []
    disagreement_columns: list[str] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]
        x_train, x_valid = _transform_pair_with_stateful_blocks(x_train, x_valid, stateful_blocks)

        train_mask = hard_mask.iloc[train_idx].to_numpy(dtype=bool)
        valid_mask = hard_mask.iloc[valid_idx].to_numpy(dtype=bool)
        train_rows = int(np.sum(train_mask))
        valid_rows = int(np.sum(valid_mask))
        if train_rows < MIN_MASKED_TRAIN_ROWS:
            raise ValueError(f"Fold {fold_number} masked train rows {train_rows} below minimum {MIN_MASKED_TRAIN_ROWS}")
        if valid_rows < MIN_MASKED_VALID_ROWS:
            raise ValueError(f"Fold {fold_number} masked valid rows {valid_rows} below minimum {MIN_MASKED_VALID_ROWS}")
        if y_train.iloc[train_mask].nunique() < 2 or y_valid.iloc[valid_mask].nunique() < 2:
            raise ValueError(f"Fold {fold_number} hard-example mask lost one class in train or valid")

        train_reference = reference_pred.iloc[train_idx]
        valid_reference = reference_pred.iloc[valid_idx]
        train_components = component_frame.iloc[train_idx] if component_frame is not None else None
        valid_components = component_frame.iloc[valid_idx] if component_frame is not None else None

        x_train, train_disagreement_columns = _append_reference_features(
            x_train,
            reference_pred=train_reference,
            reference_component_frame=train_components,
            include_logit=True,
        )
        x_valid, valid_disagreement_columns = _append_reference_features(
            x_valid,
            reference_pred=valid_reference,
            reference_component_frame=valid_components,
            include_logit=True,
        )
        if train_disagreement_columns != valid_disagreement_columns:
            raise RuntimeError("Teacher disagreement feature mismatch between train and valid folds")
        if not disagreement_columns:
            disagreement_columns = list(train_disagreement_columns)

        cat_columns = infer_categorical_columns(x_train)
        y_train_residual = y_train.iloc[train_mask].astype("float64") - train_reference.iloc[train_mask].astype("float64")
        y_valid_residual = y_valid.iloc[valid_mask].astype("float64") - valid_reference.iloc[valid_mask].astype("float64")

        model = CatBoostRegressor(
            iterations=reranker_params.iterations,
            learning_rate=reranker_params.learning_rate,
            depth=reranker_params.depth,
            l2_leaf_reg=reranker_params.l2_leaf_reg,
            random_seed=int(random_state) + fold_number,
            loss_function="RMSE",
            eval_metric="RMSE",
            verbose=verbose,
        )
        model.fit(
            x_train.iloc[train_mask],
            y_train_residual,
            cat_features=cat_columns,
            eval_set=(x_valid.iloc[valid_mask], y_valid_residual),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        valid_residual_pred = pd.Series(
            model.predict(x_valid.iloc[valid_mask]),
            index=x_valid.iloc[valid_mask].index,
            dtype="float64",
        )
        residual_pred.iloc[np.asarray(valid_idx)[valid_mask]] = valid_residual_pred.values
        final_iterations = best_iteration_or_default(model, reranker_params.iterations)
        fold_iterations.append(int(final_iterations))

        reference_auc = binary_auc(y_valid.iloc[valid_mask], valid_reference.iloc[valid_mask])
        reranked_valid = _clip_probability(valid_reference.iloc[valid_mask] + valid_residual_pred)
        reranked_auc = binary_auc(y_valid.iloc[valid_mask], reranked_valid)
        fold_rows.append(
            {
                "fold": int(fold_number),
                "masked_train_rows": train_rows,
                "masked_valid_rows": valid_rows,
                "masked_positive_rate_train": float(y_train.iloc[train_mask].mean()),
                "masked_positive_rate_valid": float(y_valid.iloc[valid_mask].mean()),
                "reference_auc_on_mask": reference_auc,
                "reranked_auc_on_mask": reranked_auc,
                "delta_auc_on_mask": float(reranked_auc - reference_auc),
                "final_iterations": int(final_iterations),
            }
        )

    if residual_pred.loc[hard_mask].isna().any():
        raise RuntimeError("Residual OOF predictions contain missing values inside the hard-example mask")

    alpha_scan_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        alpha_value = float(alpha)
        candidate = reference_pred.copy()
        candidate.loc[hard_mask] = _clip_probability(
            reference_pred.loc[hard_mask] + alpha_value * residual_pred.loc[hard_mask]
        ).values
        alpha_scan_rows.append(
            {
                "alpha": alpha_value,
                "oof_auc": float(binary_auc(y, candidate)),
                "oof_auc_on_mask": float(binary_auc(y.loc[hard_mask], candidate.loc[hard_mask])),
                "reference_auc_on_mask": float(binary_auc(y.loc[hard_mask], reference_pred.loc[hard_mask])),
            }
        )

    best_alpha_row = max(alpha_scan_rows, key=lambda row: row["oof_auc"])
    best_alpha = float(best_alpha_row["alpha"])
    candidate_best = reference_pred.copy()
    candidate_best.loc[hard_mask] = _clip_probability(
        reference_pred.loc[hard_mask] + best_alpha * residual_pred.loc[hard_mask]
    ).values

    x_full = _transform_single_with_stateful_blocks(x, stateful_blocks)
    x_full, disagreement_columns_full = _append_reference_features(
        x_full,
        reference_pred=reference_pred,
        reference_component_frame=component_frame,
        include_logit=True,
    )
    if disagreement_columns and disagreement_columns_full != disagreement_columns:
        raise RuntimeError("Teacher disagreement feature mismatch between CV folds and full train")
    if not disagreement_columns:
        disagreement_columns = list(disagreement_columns_full)

    full_iterations = max(int(round(float(np.mean(fold_iterations)))), 1)
    cat_columns = infer_categorical_columns(x_full)
    full_model = CatBoostRegressor(
        iterations=full_iterations,
        learning_rate=reranker_params.learning_rate,
        depth=reranker_params.depth,
        l2_leaf_reg=reranker_params.l2_leaf_reg,
        random_seed=reranker_params.random_seed,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=verbose,
    )
    full_model.fit(
        x_full.loc[hard_mask],
        (y.loc[hard_mask].astype("float64") - reference_pred.loc[hard_mask].astype("float64")),
        cat_features=cat_columns,
        verbose=verbose,
    )
    model_out_path = ensure_parent_dir(model_path)
    save_model(full_model, model_out_path)

    family_frame = build_family_frame(train_df)
    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "reference_pred": reference_pred.values,
            "candidate_pred": candidate_best.values,
            "residual_pred": residual_pred.values,
            "specialist_mask": hard_mask.astype("int8").values,
            family_level: family_frame[family_level].astype(str).values,
        }
    )
    oof_output = pd.concat([oof_output, stability_frame.reset_index(drop=True)], axis=1)
    oof_out_path = ensure_parent_dir(oof_path)
    oof_output.to_csv(oof_out_path, index=False)

    metrics: dict[str, Any] = {
        "approach": "hard_example_stability_residual",
        "train_rows": int(len(train_df)),
        "stability_feature_blocks": list(stability_blocks_norm),
        "reranker_feature_blocks": list(reranker_blocks_norm),
        "stability_metadata": stability_meta,
        "mask_summary": mask_summary,
        "masked_positive_rate": float(y.loc[hard_mask].mean()),
        "feature_count": int(x_full.shape[1]),
        "feature_columns": list(x_full.columns),
        "teacher_disagreement_columns": list(disagreement_columns),
        "categorical_columns": cat_columns,
        "cv_folds": int(folds),
        "stability_repeats": int(stability_repeats),
        "cv_fold_metrics": fold_rows,
        "fold_final_iterations": [int(value) for value in fold_iterations],
        "final_iterations": int(full_iterations),
        "alpha_scan": alpha_scan_rows,
        "best_alpha": best_alpha,
        "reference_oof_auc": float(binary_auc(y, reference_pred)),
        "reference_oof_auc_on_mask": float(binary_auc(y.loc[hard_mask], reference_pred.loc[hard_mask])),
        "candidate_oof_auc": float(binary_auc(y, candidate_best)),
        "candidate_oof_auc_on_mask": float(binary_auc(y.loc[hard_mask], candidate_best.loc[hard_mask])),
        "delta_vs_reference_oof_auc": float(binary_auc(y, candidate_best) - binary_auc(y, reference_pred)),
        "model_path": str(model_out_path),
        "oof_path": str(oof_out_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
    }
    write_json(metrics_path, metrics)
    return metrics
