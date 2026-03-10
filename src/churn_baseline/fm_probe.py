"""River FM/FFM probes for orthogonal tabular signals."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from river import facto, optim
from sklearn.model_selection import StratifiedKFold

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import infer_categorical_columns, load_csv, prepare_train_features
from .evaluation import binary_auc
from .feature_engineering import normalize_feature_blocks

FM = "fm"
FFM = "ffm"
FM_PROBE_FAMILIES: tuple[str, ...] = (FM, FFM)


def list_fm_probe_families() -> tuple[str, ...]:
    """Return supported factorization-model families."""
    return FM_PROBE_FAMILIES


def _save_pickle(path: str | Path, payload: Any) -> Path:
    out_path = ensure_parent_dir(path)
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return out_path


def _build_fm_model(
    *,
    model_family: str,
    n_factors: int,
    weight_lr: float,
    latent_lr: float,
    l2_weight: float,
    l2_latent: float,
    sample_normalization: bool,
    seed: int,
) -> Any:
    family = str(model_family).strip().lower()
    common_kwargs = {
        "n_factors": int(n_factors),
        "weight_optimizer": optim.AdaGrad(lr=float(weight_lr)),
        "latent_optimizer": optim.AdaGrad(lr=float(latent_lr)),
        "l2_weight": float(l2_weight),
        "l2_latent": float(l2_latent),
        "sample_normalization": bool(sample_normalization),
        "seed": int(seed),
    }
    if family == FM:
        return facto.FMClassifier(**common_kwargs)
    if family == FFM:
        return facto.FFMClassifier(**common_kwargs)
    raise ValueError(f"Unsupported model_family '{model_family}'. Available: {FM_PROBE_FAMILIES}")


def _prepare_fold_frames(
    frame: pd.DataFrame,
    *,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
    medians: pd.Series,
    means: pd.Series,
    stds: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_frame = frame.loc[:, list(categorical_columns)].copy() if categorical_columns else pd.DataFrame(index=frame.index)
    if categorical_columns:
        categorical_frame = categorical_frame.fillna("__MISSING__").astype(str)

    numeric_frame = frame.loc[:, list(numeric_columns)].copy() if numeric_columns else pd.DataFrame(index=frame.index)
    if numeric_columns:
        numeric_frame = numeric_frame.fillna(medians)
        numeric_frame = (numeric_frame - means) / stds
        numeric_frame = numeric_frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return categorical_frame, numeric_frame


def _iter_feature_dicts(
    categorical_frame: pd.DataFrame,
    numeric_frame: pd.DataFrame,
) -> Sequence[dict[str, Any]]:
    cat_columns = list(categorical_frame.columns)
    num_columns = list(numeric_frame.columns)
    cat_keys = [f"f{i:03d}_{column}" for i, column in enumerate(cat_columns)]
    num_offset = len(cat_keys)
    num_keys = [f"f{num_offset + i:03d}_{column}" for i, column in enumerate(num_columns)]

    cat_values = categorical_frame.to_numpy(dtype=object, copy=False) if cat_columns else np.empty((len(categorical_frame), 0), dtype=object)
    num_values = numeric_frame.to_numpy(dtype="float64", copy=False) if num_columns else np.empty((len(numeric_frame), 0), dtype="float64")

    for row_idx in range(len(categorical_frame.index)):
        payload: dict[str, Any] = {}
        for col_idx, key in enumerate(cat_keys):
            payload[key] = str(cat_values[row_idx, col_idx])
        for col_idx, key in enumerate(num_keys):
            value = float(num_values[row_idx, col_idx])
            if value != 0.0:
                payload[key] = value
        yield payload


def _predict_positive_probability(model: Any, features: dict[str, Any]) -> float:
    proba = model.predict_proba_one(features)
    return float(proba.get(True, proba.get(1, 0.0)))


def run_fm_probe_cv(
    *,
    train_csv_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
    feature_blocks: Sequence[str] | None,
    folds: int,
    random_state: int,
    model_family: str,
    n_factors: int,
    epochs: int,
    weight_lr: float,
    latent_lr: float,
    l2_weight: float,
    l2_latent: float,
    sample_normalization: bool,
    reference_pred: pd.Series | None = None,
    alpha_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Train a River FM/FFM probe with OOF evaluation and optional blend scan."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    train_df = load_csv(train_csv_path)
    normalized_blocks = normalize_feature_blocks(feature_blocks)
    x, y = prepare_train_features(train_df, drop_id=True, feature_blocks=normalized_blocks)
    categorical_columns = infer_categorical_columns(x)
    numeric_columns = [column for column in x.columns if column not in categorical_columns]

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_pred = pd.Series(index=x.index, dtype="float64", name="oof_pred")
    fold_rows: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx].astype(bool)
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        medians = x_train.loc[:, numeric_columns].median() if numeric_columns else pd.Series(dtype="float64")
        means = x_train.loc[:, numeric_columns].mean() if numeric_columns else pd.Series(dtype="float64")
        stds = x_train.loc[:, numeric_columns].std(ddof=0).replace(0.0, 1.0) if numeric_columns else pd.Series(dtype="float64")

        cat_train, num_train = _prepare_fold_frames(
            x_train,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            medians=medians,
            means=means,
            stds=stds,
        )
        cat_valid, num_valid = _prepare_fold_frames(
            x_valid,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            medians=medians,
            means=means,
            stds=stds,
        )

        model = _build_fm_model(
            model_family=model_family,
            n_factors=n_factors,
            weight_lr=weight_lr,
            latent_lr=latent_lr,
            l2_weight=l2_weight,
            l2_latent=l2_latent,
            sample_normalization=sample_normalization,
            seed=random_state,
        )

        train_target = y_train.tolist()
        for _ in range(int(epochs)):
            for features, target in zip(_iter_feature_dicts(cat_train, num_train), train_target):
                model.learn_one(features, target)

        valid_predictions = [
            _predict_positive_probability(model, features)
            for features in _iter_feature_dicts(cat_valid, num_valid)
        ]
        fold_pred = pd.Series(valid_predictions, index=x_valid.index, dtype="float64")
        oof_pred.iloc[valid_idx] = fold_pred.values
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "auc": float(binary_auc(y_valid, fold_pred)),
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError("OOF predictions contain missing values for FM probe.")

    full_medians = x.loc[:, numeric_columns].median() if numeric_columns else pd.Series(dtype="float64")
    full_means = x.loc[:, numeric_columns].mean() if numeric_columns else pd.Series(dtype="float64")
    full_stds = x.loc[:, numeric_columns].std(ddof=0).replace(0.0, 1.0) if numeric_columns else pd.Series(dtype="float64")
    cat_full, num_full = _prepare_fold_frames(
        x,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        medians=full_medians,
        means=full_means,
        stds=full_stds,
    )
    full_model = _build_fm_model(
        model_family=model_family,
        n_factors=n_factors,
        weight_lr=weight_lr,
        latent_lr=latent_lr,
        l2_weight=l2_weight,
        l2_latent=l2_latent,
        sample_normalization=sample_normalization,
        seed=random_state,
    )
    full_target = y.astype(bool).tolist()
    for _ in range(int(epochs)):
        for features, target in zip(_iter_feature_dicts(cat_full, num_full), full_target):
            full_model.learn_one(features, target)

    _save_pickle(
        model_path,
        {
            "model_family": str(model_family),
            "model": full_model,
            "feature_blocks": list(normalized_blocks),
            "categorical_columns": list(categorical_columns),
            "numeric_columns": list(numeric_columns),
            "medians": full_medians.to_dict(),
            "means": full_means.to_dict(),
            "stds": full_stds.to_dict(),
        },
    )

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "oof_pred": oof_pred.values,
        }
    )
    ensure_parent_dir(oof_path)
    oof_output.to_csv(oof_path, index=False)

    metrics: dict[str, Any] = {
        "train_rows": int(len(train_df)),
        "feature_count": int(x.shape[1]),
        "feature_blocks": list(normalized_blocks),
        "categorical_columns": list(categorical_columns),
        "numeric_columns": list(numeric_columns),
        "cv_folds": int(folds),
        "cv_fold_metrics": fold_rows,
        "cv_mean_auc": float(np.mean([row["auc"] for row in fold_rows])),
        "cv_std_auc": float(np.std([row["auc"] for row in fold_rows])),
        "oof_auc": float(binary_auc(y, oof_pred)),
        "model_path": str(model_path),
        "oof_path": str(oof_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "model_family": str(model_family),
        "params": {
            "n_factors": int(n_factors),
            "epochs": int(epochs),
            "weight_lr": float(weight_lr),
            "latent_lr": float(latent_lr),
            "l2_weight": float(l2_weight),
            "l2_latent": float(l2_latent),
            "sample_normalization": bool(sample_normalization),
        },
    }

    if reference_pred is not None:
        reference_by_id = pd.Series(reference_pred, copy=True)
        aligned_reference = train_df[ID_COLUMN].map(reference_by_id)
        if aligned_reference.isna().any():
            missing_ids = train_df.loc[aligned_reference.isna(), ID_COLUMN].head(5).tolist()
            raise ValueError(f"reference_pred missing ids from train: {missing_ids}")
        aligned_reference = pd.Series(aligned_reference.values, index=x.index, dtype="float64")
        metrics["reference_oof_auc"] = float(binary_auc(y, aligned_reference))
        metrics["pearson_corr_vs_reference"] = float(aligned_reference.corr(oof_pred))
        metrics["spearman_corr_vs_reference"] = float(aligned_reference.corr(oof_pred, method="spearman"))

        if alpha_grid:
            alpha_rows: list[dict[str, Any]] = []
            for alpha in alpha_grid:
                alpha_value = float(alpha)
                candidate = (1.0 - alpha_value) * aligned_reference + alpha_value * oof_pred
                alpha_rows.append(
                    {
                        "alpha": alpha_value,
                        "oof_auc": float(binary_auc(y, candidate)),
                    }
                )
            best_alpha_row = max(alpha_rows, key=lambda row: row["oof_auc"])
            metrics["blend_scan"] = alpha_rows
            metrics["best_blend_alpha"] = float(best_alpha_row["alpha"])
            metrics["best_blend_oof_auc"] = float(best_alpha_row["oof_auc"])
            metrics["delta_best_blend_vs_reference"] = float(
                best_alpha_row["oof_auc"] - binary_auc(y, aligned_reference)
            )

    write_json(metrics_path, metrics)
    return metrics
