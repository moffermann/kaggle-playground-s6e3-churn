"""Bi-gram target-encoding XGBoost experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import encode_target, infer_categorical_columns, load_csv
from .evaluation import binary_auc


DEFAULT_NGRAM_BASE_COLUMNS: tuple[str, ...] = (
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
)


@dataclass(frozen=True)
class NgramXgbParams:
    """Minimal XGBoost hyperparameters for the ngram+TE line."""

    n_estimators: int = 1500
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: float = 4.0
    subsample: float = 0.8
    colsample_bytree: float = 0.6
    reg_alpha: float = 1.0
    reg_lambda: float = 4.0
    gamma: float = 0.0
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 100

    def to_xgb_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by XGBClassifier."""
        payload = asdict(self)
        payload.update(
            {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "verbosity": 0,
            }
        )
        return payload


def _clip_probability(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    return np.clip(values.astype("float64"), epsilon, 1.0 - epsilon)


def _sanitize_total_charges(frame: pd.DataFrame, fallback_median: float | None = None) -> pd.DataFrame:
    out = frame.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        median = float(out["TotalCharges"].median()) if fallback_median is None else float(fallback_median)
        out["TotalCharges"] = out["TotalCharges"].fillna(median)
    return out


def _prepare_original_frame(original_df: pd.DataFrame, fallback_median: float) -> pd.DataFrame:
    out = original_df.copy()
    if "customerID" in out.columns and ID_COLUMN not in out.columns:
        out = out.drop(columns=["customerID"])
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = encode_target(out[TARGET_COLUMN]).astype("int8")
    out = _sanitize_total_charges(out, fallback_median=fallback_median)
    return out


def _drop_exact_original_overlaps(
    original_df: pd.DataFrame,
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    common_columns = sorted(
        (
            set(original_df.columns)
            & set(train_df.columns)
            & set(test_df.columns)
        )
        - {ID_COLUMN, TARGET_COLUMN}
    )
    if not common_columns:
        return original_df.copy(), 0

    def _row_signature(frame: pd.DataFrame) -> pd.Series:
        normalized = frame.loc[:, common_columns].copy()
        for column in common_columns:
            normalized[column] = normalized[column].fillna("__nan__").astype(str)
        return pd.util.hash_pandas_object(normalized, index=False)

    competition_signatures = pd.concat(
        [_row_signature(train_df), _row_signature(test_df)],
        axis=0,
        ignore_index=True,
    )
    overlap_mask = _row_signature(original_df).isin(set(competition_signatures.tolist()))
    filtered = original_df.loc[~overlap_mask].reset_index(drop=True)
    return filtered, int(overlap_mask.sum())


def _normalize_categorical_string(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        values = out[column]
        if pd.api.types.is_categorical_dtype(values):
            values = values.astype("object")
        out[column] = values.fillna("__nan__").astype(str)
    return out


def _build_ngram_feature_names(
    base_columns: Sequence[str],
    *,
    include_trigrams: bool,
) -> list[tuple[str, tuple[str, ...]]]:
    ngrams: list[tuple[str, tuple[str, ...]]] = []
    for cols in combinations(base_columns, 2):
        ngrams.append((f"BG_{'_'.join(cols)}", tuple(cols)))
    if include_trigrams:
        tri_base = list(base_columns)[:4]
        for cols in combinations(tri_base, 3):
            ngrams.append((f"TG_{'_'.join(cols)}", tuple(cols)))
    return ngrams


def _append_ngram_columns(frame: pd.DataFrame, specs: Sequence[tuple[str, tuple[str, ...]]]) -> pd.DataFrame:
    out = frame.copy()
    for name, parts in specs:
        values = out.loc[:, list(parts)].astype(str)
        out[name] = values.agg("__".join, axis=1)
    return out


def _resolve_te_columns(
    train_df: pd.DataFrame,
    *,
    ngram_base_columns: Sequence[str] | None = None,
    include_trigrams: bool = False,
) -> tuple[list[str], list[str], list[tuple[str, tuple[str, ...]]]]:
    categorical_columns = infer_categorical_columns(train_df.drop(columns=[TARGET_COLUMN], errors="ignore"))
    numeric_columns = [
        column
        for column in train_df.columns
        if column not in {ID_COLUMN, TARGET_COLUMN} and pd.api.types.is_numeric_dtype(train_df[column])
    ]
    ngram_base = [column for column in (ngram_base_columns or DEFAULT_NGRAM_BASE_COLUMNS) if column in categorical_columns]
    ngram_specs = _build_ngram_feature_names(ngram_base, include_trigrams=include_trigrams)
    return numeric_columns, categorical_columns, ngram_specs


def _smoothed_mean_map(
    values: pd.Series,
    target: pd.Series,
    *,
    smoothing_alpha: float,
) -> tuple[pd.Series, pd.Series, float]:
    fit_frame = pd.DataFrame({"value": values.astype(str), "target": target.astype("float64")})
    grouped = fit_frame.groupby("value", observed=False)["target"].agg(["mean", "count"])
    prior = float(fit_frame["target"].mean())
    smooth = (grouped["mean"] * grouped["count"] + prior * float(smoothing_alpha)) / (
        grouped["count"] + float(smoothing_alpha)
    )
    count_log = np.log1p(grouped["count"].astype("float64"))
    return smooth.astype("float64"), count_log.astype("float64"), prior


def _concat_fit_inputs(
    train_values: pd.Series,
    y_train: pd.Series,
    original_values: pd.Series | None,
    original_target: pd.Series | None,
) -> tuple[pd.Series, pd.Series]:
    if original_values is None or original_target is None or len(original_values) == 0:
        return train_values.reset_index(drop=True), y_train.reset_index(drop=True)
    fit_values = pd.concat([train_values.reset_index(drop=True), original_values.reset_index(drop=True)], axis=0)
    fit_target = pd.concat([y_train.reset_index(drop=True), original_target.reset_index(drop=True)], axis=0)
    return fit_values, fit_target


def _encode_target_features(
    train_frame: pd.DataFrame,
    y_train: pd.Series,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    te_columns: Sequence[str],
    inner_folds: int,
    random_state: int,
    smoothing_alpha: float,
    original_frame: pd.DataFrame | None = None,
    original_target: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.DataFrame(index=train_frame.index)
    valid_encoded = pd.DataFrame(index=valid_frame.index)
    test_encoded = pd.DataFrame(index=test_frame.index)

    inner_splitter = StratifiedKFold(
        n_splits=int(inner_folds),
        shuffle=True,
        random_state=int(random_state),
    )

    for column in te_columns:
        oof_mean = np.full(len(train_frame), float(y_train.mean()), dtype="float64")
        oof_count = np.zeros(len(train_frame), dtype="float64")

        original_values = None
        if original_frame is not None and column in original_frame.columns:
            original_values = original_frame[column].astype(str)

        for fit_idx, hold_idx in inner_splitter.split(np.zeros(len(train_frame)), y_train.to_numpy(dtype="int8")):
            fit_values, fit_target = _concat_fit_inputs(
                train_frame.iloc[fit_idx][column].astype(str),
                y_train.iloc[fit_idx],
                original_values,
                original_target,
            )
            smooth_map, count_map, prior = _smoothed_mean_map(
                fit_values,
                fit_target,
                smoothing_alpha=smoothing_alpha,
            )
            hold_values = train_frame.iloc[hold_idx][column].astype(str)
            oof_mean[hold_idx] = hold_values.map(smooth_map).fillna(prior).to_numpy(dtype="float64")
            oof_count[hold_idx] = hold_values.map(count_map).fillna(0.0).to_numpy(dtype="float64")

        full_fit_values, full_fit_target = _concat_fit_inputs(
            train_frame[column].astype(str),
            y_train,
            original_values,
            original_target,
        )
        full_smooth_map, full_count_map, full_prior = _smoothed_mean_map(
            full_fit_values,
            full_fit_target,
            smoothing_alpha=smoothing_alpha,
        )

        mean_name = f"TE_{column}"
        count_name = f"TE_COUNT_{column}"

        train_encoded[mean_name] = oof_mean.astype("float32")
        train_encoded[count_name] = oof_count.astype("float32")
        valid_encoded[mean_name] = (
            valid_frame[column].astype(str).map(full_smooth_map).fillna(full_prior).astype("float32")
        )
        valid_encoded[count_name] = (
            valid_frame[column].astype(str).map(full_count_map).fillna(0.0).astype("float32")
        )
        test_encoded[mean_name] = (
            test_frame[column].astype(str).map(full_smooth_map).fillna(full_prior).astype("float32")
        )
        test_encoded[count_name] = (
            test_frame[column].astype(str).map(full_count_map).fillna(0.0).astype("float32")
        )

    return train_encoded, valid_encoded, test_encoded


def _compute_numeric_fill_values(
    source_frame: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
) -> pd.Series:
    if not numeric_columns:
        return pd.Series(dtype="float32")
    numeric_frame = source_frame.loc[:, list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    return numeric_frame.median(numeric_only=True).astype("float32")


def _finalize_model_matrix(
    source_frame: pd.DataFrame,
    encoded_frame: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
    numeric_fill_values: pd.Series,
) -> pd.DataFrame:
    numeric_frame = source_frame.loc[:, list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if not numeric_frame.empty:
        numeric_frame = numeric_frame.fillna(numeric_fill_values).astype("float32")
    return pd.concat([numeric_frame.reset_index(drop=True), encoded_frame.reset_index(drop=True)], axis=1)


def _train_fold_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    params: NgramXgbParams,
    verbose_eval: int | bool,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**params.to_xgb_kwargs())
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=verbose_eval,
    )
    return model


def train_ngram_xgb_cv(
    *,
    train_csv_path: str | Path,
    test_csv_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
    test_pred_path: str | Path | None = None,
    original_csv_path: str | Path | None = None,
    params: NgramXgbParams | None = None,
    folds: int = 2,
    inner_folds: int = 3,
    random_state: int = 42,
    smoothing_alpha: float = 20.0,
    include_trigrams: bool = False,
    ngram_base_columns: Sequence[str] | None = None,
    verbose_eval: int | bool = False,
) -> dict[str, Any]:
    """Train the minimal ngram+TE XGBoost line with outer CV."""
    params = params or NgramXgbParams(random_state=random_state)
    train_df = load_csv(train_csv_path)
    test_df = load_csv(test_csv_path)
    train_df = _sanitize_total_charges(train_df)
    test_df = _sanitize_total_charges(test_df, fallback_median=float(train_df["TotalCharges"].median()))
    y = encode_target(train_df[TARGET_COLUMN]).astype("int8")

    original_df: pd.DataFrame | None = None
    original_target: pd.Series | None = None
    if original_csv_path:
        original_raw = load_csv(original_csv_path)
        original_df = _prepare_original_frame(original_raw, fallback_median=float(train_df["TotalCharges"].median()))
        if TARGET_COLUMN not in original_df.columns:
            raise ValueError(f"Original dataset at {original_csv_path} must contain '{TARGET_COLUMN}'.")
        original_df, original_overlap_rows_removed = _drop_exact_original_overlaps(
            original_df,
            train_df=train_df,
            test_df=test_df,
        )
        original_target = original_df[TARGET_COLUMN].astype("int8")
    else:
        original_overlap_rows_removed = 0

    numeric_columns, categorical_columns, ngram_specs = _resolve_te_columns(
        train_df,
        ngram_base_columns=ngram_base_columns,
        include_trigrams=include_trigrams,
    )

    train_source = train_df.drop(columns=[TARGET_COLUMN]).copy()
    test_source = test_df.copy()
    original_source = None
    if original_df is not None:
        original_source = original_df.drop(columns=[TARGET_COLUMN], errors="ignore").copy()

    frames_to_augment = [train_source, test_source]
    if original_source is not None:
        frames_to_augment.append(original_source)
    frames_to_augment = [_normalize_categorical_string(frame, categorical_columns) for frame in frames_to_augment]
    frames_to_augment = [_append_ngram_columns(frame, ngram_specs) for frame in frames_to_augment]
    train_source, test_source, *rest = frames_to_augment
    if rest:
        original_source = rest[0]

    te_columns = list(categorical_columns) + [name for name, _ in ngram_specs]

    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(random_state))
    oof_pred = np.zeros(len(train_df), dtype="float64")
    fold_aucs: list[float] = []
    fold_best_iterations: list[int] = []
    test_pred = np.zeros(len(test_df), dtype="float64")

    for fold_idx, (fit_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(train_df)), y.to_numpy(dtype="int8")), start=1):
        fit_source = train_source.iloc[fit_idx].reset_index(drop=True)
        valid_source = train_source.iloc[valid_idx].reset_index(drop=True)
        y_fit = y.iloc[fit_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)
        numeric_fill_values = _compute_numeric_fill_values(fit_source, numeric_columns=numeric_columns)

        fit_encoded, valid_encoded, test_encoded = _encode_target_features(
            fit_source,
            y_fit,
            valid_source,
            test_source.reset_index(drop=True),
            te_columns=te_columns,
            inner_folds=inner_folds,
            random_state=int(random_state) + fold_idx,
            smoothing_alpha=smoothing_alpha,
            original_frame=original_source,
            original_target=original_target,
        )

        x_fit = _finalize_model_matrix(
            fit_source,
            fit_encoded,
            numeric_columns=numeric_columns,
            numeric_fill_values=numeric_fill_values,
        )
        x_valid = _finalize_model_matrix(
            valid_source,
            valid_encoded,
            numeric_columns=numeric_columns,
            numeric_fill_values=numeric_fill_values,
        )
        x_test = _finalize_model_matrix(
            test_source.reset_index(drop=True),
            test_encoded,
            numeric_columns=numeric_columns,
            numeric_fill_values=numeric_fill_values,
        )

        model = _train_fold_model(
            x_fit,
            y_fit,
            x_valid,
            y_valid,
            params=params,
            verbose_eval=verbose_eval,
        )
        valid_pred = _clip_probability(model.predict_proba(x_valid)[:, 1])
        test_fold_pred = _clip_probability(model.predict_proba(x_test)[:, 1])

        oof_pred[valid_idx] = valid_pred
        test_pred += test_fold_pred / float(folds)
        fold_auc = float(binary_auc(y_valid, valid_pred))
        fold_aucs.append(fold_auc)
        best_iter = getattr(model, "best_iteration", None)
        fold_best_iterations.append(int(best_iter) if best_iter is not None else int(params.n_estimators))

    oof_auc = float(binary_auc(y, oof_pred))
    metrics = {
        "experiment_name": "ngram_te_xgb",
        "model_family": "xgboost",
        "train_csv_path": str(train_csv_path),
        "test_csv_path": str(test_csv_path),
        "original_csv_path": str(original_csv_path) if original_csv_path else None,
        "used_original": bool(original_source is not None),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "original_rows": int(len(original_source)) if original_source is not None else 0,
        "original_overlap_rows_removed": int(original_overlap_rows_removed),
        "folds": int(folds),
        "inner_folds": int(inner_folds),
        "random_state": int(random_state),
        "smoothing_alpha": float(smoothing_alpha),
        "include_trigrams": bool(include_trigrams),
        "numeric_columns": list(numeric_columns),
        "categorical_columns": list(categorical_columns),
        "ngram_base_columns": [column for column in (ngram_base_columns or DEFAULT_NGRAM_BASE_COLUMNS) if column in categorical_columns],
        "ngram_columns": [name for name, _ in ngram_specs],
        "te_columns": list(te_columns),
        "oof_auc": oof_auc,
        "cv_mean_auc": float(np.mean(fold_aucs)),
        "cv_std_auc": float(np.std(fold_aucs, ddof=0)),
        "fold_aucs": [float(value) for value in fold_aucs],
        "fold_best_iterations": [int(value) for value in fold_best_iterations],
        "params": params.to_xgb_kwargs(),
        "oof_path": str(oof_path),
        "metrics_path": str(metrics_path),
    }
    if test_pred_path:
        metrics["test_pred_path"] = str(test_pred_path)

    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].astype("int64"),
            "target": y.astype("int8"),
            "oof_pred": _clip_probability(oof_pred),
        }
    )
    oof_out = ensure_parent_dir(oof_path)
    oof_frame.to_csv(oof_out, index=False)

    if test_pred_path:
        test_out = ensure_parent_dir(test_pred_path)
        pd.DataFrame(
            {
                ID_COLUMN: test_df[ID_COLUMN].astype("int64"),
                TARGET_COLUMN: _clip_probability(test_pred),
            }
        ).to_csv(test_out, index=False)

    write_json(metrics_path, metrics)
    return metrics
