"""Sparse linear probes for orthogonal tabular signals."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import encode_target, infer_categorical_columns, load_csv, prepare_train_features
from .evaluation import binary_auc
from .feature_engineering import normalize_feature_blocks
from .specialist import _append_teacher_disagreement_features, _normalize_reference_component_frame, _prob_to_logit

LOGISTIC = "logistic"
SPLINE_LOGISTIC = "spline_logistic"
CATBOOST_META = "catboost_meta"
LINEAR_PROBE_FAMILIES: tuple[str, ...] = (LOGISTIC, SPLINE_LOGISTIC, CATBOOST_META)
FEATURE_MODE_RAW = "raw"
FEATURE_MODE_TEACHER_META = "teacher_meta"
LINEAR_PROBE_FEATURE_MODES: tuple[str, ...] = (FEATURE_MODE_RAW, FEATURE_MODE_TEACHER_META)
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


def list_linear_probe_feature_modes() -> tuple[str, ...]:
    """Return supported feature modes for the linear probe."""
    return LINEAR_PROBE_FEATURE_MODES


def _build_tenure_bin(tenure_values: pd.Series) -> pd.Series:
    bins = [-np.inf, 6, 12, 24, 48, np.inf]
    labels = ["0_6", "7_12", "13_24", "25_48", "49_plus"]
    return pd.cut(tenure_values, bins=bins, labels=labels).astype(str)


def _prepare_teacher_meta_features(
    *,
    train_df: pd.DataFrame,
    reference_pred: pd.Series,
    teacher_component_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build a compact meta-model matrix from teacher components plus cohort keys."""
    cohort = pd.DataFrame(
        {
            "PaymentMethod": train_df["PaymentMethod"].astype(str),
            "Contract": train_df["Contract"].astype(str),
            "InternetService": train_df["InternetService"].astype(str),
            "PaperlessBilling": train_df["PaperlessBilling"].astype(str),
            "tenure_bin": _build_tenure_bin(pd.to_numeric(train_df["tenure"], errors="coerce").fillna(0.0)),
        },
        index=train_df.index,
    )
    y = encode_target(train_df[TARGET_COLUMN])
    aligned_reference = _align_reference_prediction(
        train_ids=train_df[ID_COLUMN],
        reference_pred=reference_pred,
        index=cohort.index,
    )
    component_frame = _normalize_reference_component_frame(train_df[ID_COLUMN], teacher_component_frame)

    x = cohort.copy()
    x["reference_pred_feature"] = aligned_reference.values
    x["reference_logit_feature"] = _prob_to_logit(aligned_reference)
    x, disagreement_columns = _append_teacher_disagreement_features(x, aligned_reference, component_frame)
    return x, y, disagreement_columns


def _align_reference_prediction(
    *,
    train_ids: pd.Series,
    reference_pred: pd.Series,
    index: pd.Index,
) -> pd.Series:
    reference_by_id = pd.Series(reference_pred, copy=True)
    if reference_by_id.index.has_duplicates:
        duplicate_ids = pd.Index(reference_by_id.index[reference_by_id.index.duplicated()]).unique().tolist()[:5]
        raise ValueError(
            "reference_pred must be indexed by unique train ids and represent OOF predictions. "
            f"Duplicate ids found: {duplicate_ids}"
        )
    aligned_reference = train_ids.map(reference_by_id)
    if aligned_reference.isna().any():
        missing_ids = train_ids.loc[aligned_reference.isna()].head(5).tolist()
        raise ValueError(
            "reference_pred must cover every train id with OOF-aligned predictions. "
            f"Missing ids: {missing_ids}"
        )
    return pd.Series(aligned_reference.values, index=index, dtype="float64")


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


def _build_catboost_meta_model(
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    random_state: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        l2_leaf_reg=float(l2_leaf_reg),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=int(random_state),
        allow_writing_files=False,
        verbose=False,
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
    feature_mode: str = FEATURE_MODE_RAW,
    reference_pred: pd.Series | None = None,
    teacher_component_frame: pd.DataFrame | None = None,
    alpha_grid: Sequence[float] | None = None,
    reference_is_oof: bool = False,
    tree_iterations: int = 300,
    tree_depth: int = 4,
    tree_learning_rate: float = 0.05,
    tree_l2_leaf_reg: float = 8.0,
    tree_early_stopping_rounds: int = 50,
) -> dict[str, Any]:
    """Train a sparse linear probe with OOF evaluation and optional blend scan."""
    if folds < 2:
        raise ValueError("folds must be >= 2")

    train_df = load_csv(train_csv_path)
    model_family_value = str(model_family).strip().lower()
    feature_mode_value = str(feature_mode).strip().lower()
    disagreement_columns: list[str] = []
    if model_family_value == CATBOOST_META and feature_mode_value != FEATURE_MODE_TEACHER_META:
        raise ValueError("model_family=catboost_meta is only supported with feature_mode=teacher_meta")
    if model_family_value == CATBOOST_META and int(tree_early_stopping_rounds) < 1:
        raise ValueError("tree_early_stopping_rounds must be >= 1 for model_family=catboost_meta")
    if reference_pred is not None and not reference_is_oof:
        raise ValueError(
            "reference_pred must be explicit OOF data. Pass reference_is_oof=True only after "
            "verifying the reference predictions and teacher components are OOF-aligned."
        )
    if feature_mode_value == FEATURE_MODE_RAW:
        normalized_blocks = normalize_feature_blocks(feature_blocks)
        x, y = prepare_train_features(train_df, drop_id=True, feature_blocks=normalized_blocks)
    elif feature_mode_value == FEATURE_MODE_TEACHER_META:
        if reference_pred is None:
            raise ValueError("feature_mode=teacher_meta requires reference_pred")
        if teacher_component_frame is None:
            raise ValueError("feature_mode=teacher_meta requires teacher_component_frame with pred_* columns")
        normalized_blocks = ()
        x, y, disagreement_columns = _prepare_teacher_meta_features(
            train_df=train_df,
            reference_pred=reference_pred,
            teacher_component_frame=teacher_component_frame,
        )
    else:
        raise ValueError(f"Unsupported feature_mode '{feature_mode}'. Available: {LINEAR_PROBE_FEATURE_MODES}")
    categorical_columns = infer_categorical_columns(x)
    numeric_columns = [column for column in x.columns if column not in categorical_columns]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    oof_pred = pd.Series(index=x.index, dtype="float64", name="oof_pred")
    fold_rows: list[dict[str, Any]] = []
    fold_iterations: list[int] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        if model_family_value == CATBOOST_META:
            fold_model = _build_catboost_meta_model(
                iterations=tree_iterations,
                learning_rate=tree_learning_rate,
                depth=tree_depth,
                l2_leaf_reg=tree_l2_leaf_reg,
                random_state=random_state,
            )
            fold_model.fit(
                x_train,
                y_train,
                cat_features=list(categorical_columns),
                eval_set=(x_valid, y_valid),
                use_best_model=True,
                early_stopping_rounds=int(tree_early_stopping_rounds),
                verbose=False,
            )
            fold_pred = pd.Series(
                fold_model.predict_proba(x_valid)[:, 1],
                index=x_valid.index,
                dtype="float64",
            )
            best_iter = fold_model.get_best_iteration()
            final_iter = int(best_iter) + 1 if best_iter is not None and best_iter >= 0 else int(tree_iterations)
            fold_iterations.append(max(final_iter, 1))
        else:
            fold_model = _build_linear_pipeline(
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
            fold_model.fit(x_train, y_train)
            fold_pred = pd.Series(
                fold_model.predict_proba(x_valid)[:, 1],
                index=x_valid.index,
                dtype="float64",
            )
            best_iter = None
            final_iter = None
        oof_pred.iloc[valid_idx] = fold_pred.values
        fold_row = {
            "fold": int(fold_number),
            "valid_rows": int(len(valid_idx)),
            "auc": float(binary_auc(y_valid, fold_pred)),
        }
        if best_iter is not None:
            fold_row["best_iteration"] = int(best_iter)
            fold_row["final_iterations"] = int(max(final_iter or 1, 1))
        fold_rows.append(fold_row)

    if oof_pred.isna().any():
        raise RuntimeError("OOF predictions contain missing values for linear probe.")

    if model_family_value == CATBOOST_META:
        full_iterations = max(int(round(float(np.mean(fold_iterations)))), 1) if fold_iterations else int(tree_iterations)
        full_model = _build_catboost_meta_model(
            iterations=full_iterations,
            learning_rate=tree_learning_rate,
            depth=tree_depth,
            l2_leaf_reg=tree_l2_leaf_reg,
            random_state=random_state,
        )
        full_model.fit(x, y, cat_features=list(categorical_columns), verbose=False)
        _save_pickle(model_path, full_model)
    else:
        full_model = _build_linear_pipeline(
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
        full_model.fit(x, y)
        _save_pickle(model_path, full_model)
        full_iterations = None

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
        "feature_mode": feature_mode_value,
        "teacher_disagreement_columns": disagreement_columns,
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
        "reference_is_oof": bool(reference_is_oof),
        "params": {
            "C": float(c_value),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "solver": "saga",
            "spline_n_knots": int(spline_n_knots),
            "spline_degree": int(spline_degree),
            "spline_columns": _resolve_spline_columns(numeric_columns) if model_family_value == SPLINE_LOGISTIC else [],
            "tree_iterations": int(tree_iterations),
            "tree_depth": int(tree_depth),
            "tree_learning_rate": float(tree_learning_rate),
            "tree_l2_leaf_reg": float(tree_l2_leaf_reg),
            "tree_early_stopping_rounds": int(tree_early_stopping_rounds),
        },
    }
    if full_iterations is not None:
        metrics["final_iterations"] = int(full_iterations)

    if reference_pred is not None:
        aligned_reference = _align_reference_prediction(
            train_ids=train_df[ID_COLUMN],
            reference_pred=reference_pred,
            index=x.index,
        )
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
