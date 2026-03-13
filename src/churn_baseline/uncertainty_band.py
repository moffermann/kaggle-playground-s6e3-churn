"""Localized reranker over the incumbent uncertainty band."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold

from .artifacts import ensure_parent_dir, write_json
from .config import CatBoostHyperParams, ID_COLUMN, TARGET_COLUMN
from .data import encode_target, infer_categorical_columns, load_csv
from .diagnostics import build_family_frame
from .evaluation import binary_auc
from .pipeline import _prepare_train_matrix, _transform_pair_with_stateful_blocks, _transform_single_with_stateful_blocks
from .specialist import (
    _append_reference_features,
    _clip_probability,
    _normalize_reference_component_frame,
)


DEFAULT_V3_OOF_SPEC = "artifacts/reports/validation_protocol_v3_chain_oof.csv#candidate_pred"
MIN_BAND_TRAIN_ROWS = 2000
MIN_BAND_VALID_ROWS = 500


def _align_reference_prediction(
    train_df: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> pd.Series:
    required = {ID_COLUMN, "target", "reference_pred"}
    missing = sorted(required.difference(reference_frame.columns))
    if missing:
        raise ValueError(f"reference_frame missing required columns: {missing}")

    out = train_df[[ID_COLUMN, TARGET_COLUMN]].merge(
        reference_frame[[ID_COLUMN, "target", "reference_pred"]],
        how="left",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if out["reference_pred"].isna().any():
        missing_ids = out.loc[out["reference_pred"].isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference_frame missing ids from train.csv: {missing_ids}")

    train_target = encode_target(out[TARGET_COLUMN]).astype("int8")
    reference_target = pd.to_numeric(out["target"], errors="raise").astype("int8")
    if not np.array_equal(train_target.to_numpy(dtype="int8"), reference_target.to_numpy(dtype="int8")):
        raise ValueError("reference_frame target does not align with train.csv")

    reference_pred = pd.to_numeric(out["reference_pred"], errors="raise").astype("float64")
    if reference_pred.isna().any():
        raise ValueError("reference_pred contains NaN values after alignment")
    return pd.Series(reference_pred.values, index=train_df.index, dtype="float64", name="reference_pred")


def _build_component_stats(
    ids: pd.Series,
    reference_component_frame: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    component_frame = _normalize_reference_component_frame(ids, reference_component_frame)
    if component_frame is None:
        return None, None
    component_columns = [column for column in component_frame.columns if column != ID_COLUMN]
    component_std = pd.Series(
        component_frame[component_columns].to_numpy(dtype="float64").std(axis=1),
        index=ids.index,
        dtype="float64",
        name="teacher_component_std",
    )
    return component_frame, component_std


def build_uncertainty_band_mask(
    train_df: pd.DataFrame,
    *,
    reference_pred: pd.Series,
    reference_component_frame: pd.DataFrame | None,
    target_family_level: str,
    target_family_value: str,
    band_half_width: float,
    min_teacher_std: float | None,
    max_relative_mask_drift: float,
) -> tuple[pd.Series, dict[str, Any], pd.Series | None]:
    if target_family_level not in {"segment3", "segment5"}:
        raise ValueError("target_family_level must be 'segment3' or 'segment5'")
    if float(band_half_width) <= 0.0 or float(band_half_width) >= 0.5:
        raise ValueError(f"band_half_width must be in (0, 0.5), got {band_half_width}")
    if float(max_relative_mask_drift) < 0.0:
        raise ValueError(f"max_relative_mask_drift must be >= 0, got {max_relative_mask_drift}")

    family_frame = build_family_frame(train_df)
    if family_frame[target_family_level].isna().any():
        raise ValueError(f"family frame contains NaN values in '{target_family_level}'")

    _, component_std = _build_component_stats(train_df[ID_COLUMN], reference_component_frame)

    family_mask = family_frame[target_family_level].astype(str).eq(str(target_family_value))
    uncertainty = (reference_pred.astype("float64") - 0.5).abs()
    band_mask = uncertainty.le(float(band_half_width))
    baseline_mask = family_mask & band_mask
    final_mask = baseline_mask.copy()
    if min_teacher_std is not None:
        if component_std is None:
            raise ValueError("min_teacher_std requires reference_component_frame")
        final_mask = final_mask & component_std.ge(float(min_teacher_std))

    baseline_rows = int(baseline_mask.sum())
    final_rows = int(final_mask.sum())
    baseline_rate = float(baseline_rows / max(len(train_df), 1))
    final_rate = float(final_rows / max(len(train_df), 1))
    relative_drift = 0.0
    if baseline_rows > 0:
        relative_drift = float(abs(final_rows - baseline_rows) / baseline_rows)
    if relative_drift > float(max_relative_mask_drift):
        raise ValueError(
            "uncertainty-band mask rate drift exceeds allowed maximum: "
            f"baseline_rows={baseline_rows}, final_rows={final_rows}, relative_drift={relative_drift:.6f}, "
            f"max_relative_mask_drift={float(max_relative_mask_drift):.6f}"
        )

    mask_summary = {
        "target_family_level": str(target_family_level),
        "target_family_value": str(target_family_value),
        "band_half_width": float(band_half_width),
        "min_teacher_std": None if min_teacher_std is None else float(min_teacher_std),
        "max_relative_mask_drift": float(max_relative_mask_drift),
        "family_rows": int(family_mask.sum()),
        "family_rate": float(family_mask.mean()),
        "band_rows": int(band_mask.sum()),
        "band_rate": float(band_mask.mean()),
        "baseline_mask_rows": baseline_rows,
        "baseline_mask_rate": baseline_rate,
        "final_mask_rows": final_rows,
        "final_mask_rate": final_rate,
        "relative_mask_rate_drift": relative_drift,
        "reference_uncertainty_mean_on_mask": float(uncertainty.loc[final_mask].mean()) if final_rows else None,
        "reference_uncertainty_max_on_mask": float(uncertainty.loc[final_mask].max()) if final_rows else None,
        "teacher_component_std_mean_on_mask": (
            float(component_std.loc[final_mask].mean()) if component_std is not None and final_rows else None
        ),
    }
    return final_mask.astype(bool), mask_summary, component_std


def run_uncertainty_band_reranker_cv(
    *,
    train_csv_path: str | Path,
    reference_frame: pd.DataFrame,
    reference_component_frame: pd.DataFrame | None,
    params: CatBoostHyperParams,
    feature_blocks: Sequence[str] | None,
    folds: int,
    random_state: int,
    early_stopping_rounds: int,
    verbose: int,
    alpha_grid: Sequence[float],
    target_family_level: str,
    target_family_value: str,
    band_half_width: float,
    min_teacher_std: float | None,
    max_relative_mask_drift: float,
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one value")

    train_df = load_csv(train_csv_path)
    normalized_blocks, _, stateful_blocks, x, y = _prepare_train_matrix(train_df, feature_blocks)
    reference_pred = _align_reference_prediction(train_df, reference_frame)
    component_frame, teacher_component_std = _build_component_stats(train_df[ID_COLUMN], reference_component_frame)
    band_mask, mask_summary, teacher_component_std = build_uncertainty_band_mask(
        train_df,
        reference_pred=reference_pred,
        reference_component_frame=component_frame,
        target_family_level=target_family_level,
        target_family_value=target_family_value,
        band_half_width=band_half_width,
        min_teacher_std=min_teacher_std,
        max_relative_mask_drift=max_relative_mask_drift,
    )

    mask_rows = int(band_mask.sum())
    if mask_rows < MIN_BAND_TRAIN_ROWS:
        raise ValueError(
            f"uncertainty band selects only {mask_rows} rows. Minimum required: {MIN_BAND_TRAIN_ROWS}"
        )
    if y.loc[band_mask].nunique() < 2:
        raise ValueError("uncertainty band must contain both classes")

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

        train_mask = band_mask.iloc[train_idx].to_numpy(dtype=bool)
        valid_mask = band_mask.iloc[valid_idx].to_numpy(dtype=bool)
        masked_train_rows = int(np.sum(train_mask))
        masked_valid_rows = int(np.sum(valid_mask))
        if masked_train_rows < MIN_BAND_TRAIN_ROWS:
            raise ValueError(
                f"Fold {fold_number} train rows inside uncertainty band {masked_train_rows} below minimum {MIN_BAND_TRAIN_ROWS}"
            )
        if masked_valid_rows < MIN_BAND_VALID_ROWS:
            raise ValueError(
                f"Fold {fold_number} valid rows inside uncertainty band {masked_valid_rows} below minimum {MIN_BAND_VALID_ROWS}"
            )
        if y_train.iloc[train_mask].nunique() < 2 or y_valid.iloc[valid_mask].nunique() < 2:
            raise ValueError(f"Fold {fold_number} uncertainty band lost one class in train or valid")

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

        fold_model = CatBoostRegressor(
            iterations=params.iterations,
            learning_rate=params.learning_rate,
            depth=params.depth,
            l2_leaf_reg=params.l2_leaf_reg,
            random_seed=random_state,
            loss_function="RMSE",
            eval_metric="RMSE",
            verbose=verbose,
        )
        fold_model.fit(
            x_train.iloc[train_mask],
            y_train_residual,
            cat_features=cat_columns,
            eval_set=(x_valid.iloc[valid_mask], y_valid_residual),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        valid_residual_pred = pd.Series(
            fold_model.predict(x_valid.iloc[valid_mask]),
            index=x_valid.iloc[valid_mask].index,
            dtype="float64",
        )
        residual_pred.iloc[np.asarray(valid_idx)[valid_mask]] = valid_residual_pred.values

        fold_best_iter = fold_model.get_best_iteration()
        fold_final_iterations = params.iterations if fold_best_iter is None or fold_best_iter < 0 else int(fold_best_iter) + 1
        fold_iterations.append(int(fold_final_iterations))
        reference_auc = binary_auc(y_valid.iloc[valid_mask], valid_reference.iloc[valid_mask])
        reranked_valid = _clip_probability(valid_reference.iloc[valid_mask] + valid_residual_pred)
        reranked_auc = binary_auc(y_valid.iloc[valid_mask], reranked_valid)
        fold_rows.append(
            {
                "fold": int(fold_number),
                "masked_train_rows": masked_train_rows,
                "masked_valid_rows": masked_valid_rows,
                "masked_positive_rate_train": float(y_train.iloc[train_mask].mean()),
                "masked_positive_rate_valid": float(y_valid.iloc[valid_mask].mean()),
                "reference_auc_on_mask": reference_auc,
                "reranked_auc_on_mask": reranked_auc,
                "delta_auc_on_mask": float(reranked_auc - reference_auc),
                "final_iterations": int(fold_final_iterations),
            }
        )

    if residual_pred.loc[band_mask].isna().any():
        raise RuntimeError("Residual OOF predictions contain missing values inside the uncertainty band")

    alpha_scan_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        alpha_value = float(alpha)
        candidate = reference_pred.copy()
        candidate.loc[band_mask] = _clip_probability(
            reference_pred.loc[band_mask] + alpha_value * residual_pred.loc[band_mask]
        ).values
        alpha_scan_rows.append(
            {
                "alpha": alpha_value,
                "oof_auc": float(binary_auc(y, candidate)),
                "oof_auc_on_mask": float(binary_auc(y.loc[band_mask], candidate.loc[band_mask])),
                "reference_auc_on_mask": float(binary_auc(y.loc[band_mask], reference_pred.loc[band_mask])),
            }
        )

    best_alpha_row = max(alpha_scan_rows, key=lambda row: row["oof_auc"])
    best_alpha = float(best_alpha_row["alpha"])
    candidate_best = reference_pred.copy()
    candidate_best.loc[band_mask] = _clip_probability(
        reference_pred.loc[band_mask] + best_alpha * residual_pred.loc[band_mask]
    ).values

    final_iterations = max(int(round(float(np.mean(fold_iterations)))), 1)
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

    cat_columns = infer_categorical_columns(x_full)
    full_model = CatBoostRegressor(
        iterations=final_iterations,
        learning_rate=params.learning_rate,
        depth=params.depth,
        l2_leaf_reg=params.l2_leaf_reg,
        random_seed=params.random_seed,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=verbose,
    )
    full_model.fit(
        x_full.loc[band_mask],
        (y.loc[band_mask].astype("float64") - reference_pred.loc[band_mask].astype("float64")),
        cat_features=cat_columns,
        verbose=verbose,
    )
    model_out_path = ensure_parent_dir(model_path)
    full_model.save_model(str(model_out_path))

    family_frame = build_family_frame(train_df)
    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "reference_pred": reference_pred.values,
            "candidate_pred": candidate_best.values,
            "residual_pred": residual_pred.values,
            "family_mask": family_frame[target_family_level].astype(str).eq(str(target_family_value)).astype("int8").values,
            "uncertainty_mask": (reference_pred.sub(0.5).abs().le(float(band_half_width))).astype("int8").values,
            "specialist_mask": band_mask.astype("int8").values,
            "reference_uncertainty": (reference_pred.sub(0.5).abs()).astype("float64").values,
        }
    )
    if teacher_component_std is not None:
        oof_output["teacher_component_std"] = teacher_component_std.astype("float64").values
    oof_out_path = ensure_parent_dir(oof_path)
    oof_output.to_csv(oof_out_path, index=False)

    metrics: dict[str, Any] = {
        "approach": "uncertainty_band_residual",
        "train_rows": int(len(train_df)),
        "feature_blocks": list(normalized_blocks),
        "feature_count": int(x_full.shape[1]),
        "feature_columns": list(x_full.columns),
        "teacher_disagreement_columns": list(disagreement_columns),
        "categorical_columns": cat_columns,
        "mask_summary": mask_summary,
        "masked_positive_rate": float(y.loc[band_mask].mean()),
        "cv_folds": int(folds),
        "cv_fold_metrics": fold_rows,
        "fold_final_iterations": [int(value) for value in fold_iterations],
        "final_iterations": int(final_iterations),
        "alpha_scan": alpha_scan_rows,
        "best_alpha": best_alpha,
        "reference_oof_auc": float(binary_auc(y, reference_pred)),
        "reference_oof_auc_on_mask": float(binary_auc(y.loc[band_mask], reference_pred.loc[band_mask])),
        "candidate_oof_auc": float(binary_auc(y, candidate_best)),
        "candidate_oof_auc_on_mask": float(binary_auc(y.loc[band_mask], candidate_best.loc[band_mask])),
        "delta_vs_reference_oof_auc": float(binary_auc(y, candidate_best) - binary_auc(y, reference_pred)),
        "model_path": str(model_out_path),
        "oof_path": str(oof_out_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
    }
    write_json(metrics_path, metrics)
    return metrics
