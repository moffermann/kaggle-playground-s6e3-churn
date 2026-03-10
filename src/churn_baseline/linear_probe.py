"""Sparse linear probes for orthogonal tabular signals."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import infer_categorical_columns, load_csv, prepare_train_features
from .evaluation import binary_auc
from .feature_engineering import normalize_feature_blocks

LOGISTIC = "logistic"
SPLINE_LOGISTIC = "spline_logistic"
LINEAR_PROBE_FAMILIES: tuple[str, ...] = (LOGISTIC, SPLINE_LOGISTIC)
PREFERRED_SPLINE_COLUMNS: tuple[str, ...] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "effective_monthly_charge",
    "current_vs_effective_monthly_delta",
    "current_vs_effective_monthly_ratio",
    "monthly_per_active_service",
    "payment_friction_index",
    "months_to_contract_boundary",
    "contract_cycle_progress",
    "annual_cycle_progress",
    "two_year_cycle_progress",
    "monthly_rank_in_value_cohort",
)


def _save_pickle(path: str | Path, payload: Any) -> Path:
    out_path = ensure_parent_dir(path)
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return out_path


def list_linear_probe_families() -> tuple[str, ...]:
    """Return supported linear probe model families."""
    return LINEAR_PROBE_FAMILIES


def _resolve_spline_columns(numeric_columns: Sequence[str]) -> list[str]:
    numeric_set = set(numeric_columns)
    selected = [column for column in PREFERRED_SPLINE_COLUMNS if column in numeric_set]
    if selected:
        return selected
    return list(numeric_columns[: min(len(numeric_columns), 4)])


def _build_linear_pipeline(
    *,
    model_family: str,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
    c_value: float,
    max_iter: int,
    random_state: int,
    tol: float,
    verbose: int,
    spline_n_knots: int,
    spline_degree: int,
) -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    model_family_value = str(model_family).strip().lower()
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, list(categorical_columns)))

    if model_family_value == LOGISTIC:
        if numeric_columns:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    list(numeric_columns),
                )
            )
    elif model_family_value == SPLINE_LOGISTIC:
        spline_columns = _resolve_spline_columns(numeric_columns)
        remaining_numeric = [column for column in numeric_columns if column not in set(spline_columns)]
        if spline_columns:
            transformers.append(
                (
                    "num_spline",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                            (
                                "spline",
                                SplineTransformer(
                                    n_knots=int(spline_n_knots),
                                    degree=int(spline_degree),
                                    knots="quantile",
                                    extrapolation="constant",
                                    include_bias=False,
                                    sparse_output=True,
                                ),
                            ),
                        ]
                    ),
                    list(spline_columns),
                )
            )
        if remaining_numeric:
            transformers.append(
                (
                    "num_linear",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    list(remaining_numeric),
                )
            )
    else:
        raise ValueError(f"Unsupported model_family '{model_family}'. Available: {LINEAR_PROBE_FAMILIES}")

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    classifier = LogisticRegression(
        C=float(c_value),
        solver="saga",
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(random_state),
        verbose=int(verbose),
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", classifier),
        ]
    )


def run_linear_probe_cv(
    *,
    train_csv_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
    feature_blocks: Sequence[str] | None,
    folds: int,
    random_state: int,
    c_value: float,
    max_iter: int,
    tol: float,
    verbose: int,
    model_family: str,
    spline_n_knots: int,
    spline_degree: int,
    reference_pred: pd.Series | None = None,
    alpha_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Train a sparse linear probe with OOF evaluation and optional blend scan."""
    if folds < 2:
        raise ValueError("folds must be >= 2")

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
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        fold_pipeline = _build_linear_pipeline(
            model_family=model_family,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            c_value=c_value,
            max_iter=max_iter,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            spline_n_knots=spline_n_knots,
            spline_degree=spline_degree,
        )
        fold_pipeline.fit(x_train, y_train)
        fold_pred = pd.Series(
            fold_pipeline.predict_proba(x_valid)[:, 1],
            index=x_valid.index,
            dtype="float64",
        )
        oof_pred.iloc[valid_idx] = fold_pred.values
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "auc": float(binary_auc(y_valid, fold_pred)),
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError("OOF predictions contain missing values for linear probe.")

    full_pipeline = _build_linear_pipeline(
        model_family=model_family,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        c_value=c_value,
        max_iter=max_iter,
        random_state=random_state,
        tol=tol,
        verbose=verbose,
        spline_n_knots=spline_n_knots,
        spline_degree=spline_degree,
    )
    full_pipeline.fit(x, y)
    _save_pickle(model_path, full_pipeline)

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
            "C": float(c_value),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "solver": "saga",
            "spline_n_knots": int(spline_n_knots),
            "spline_degree": int(spline_degree),
            "spline_columns": _resolve_spline_columns(numeric_columns) if model_family == SPLINE_LOGISTIC else [],
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
