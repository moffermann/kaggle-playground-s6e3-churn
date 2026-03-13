"""External Telco teacher transfer feature experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import CatBoostHyperParams, ID_COLUMN, TARGET_COLUMN
from .data import infer_categorical_columns, load_csv, prepare_test_features, prepare_train_features
from .diagnostics import describe_file, utc_now_iso
from .evaluation import binary_auc
from .feature_engineering import normalize_feature_blocks
from .modeling import best_iteration_or_default, fit_full_train, fit_with_validation, predict_proba
from .pipeline import _build_stratify_labels, _prepare_train_matrix, _transform_pair_with_stateful_blocks


DEFAULT_ORIGINAL_CSV = "artifacts/external/blastchar_telco/WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXTERNAL_ID_COLUMN = "customerID"
TRANSFER_FEATURE_COLUMN = "external_telco_pred"
DATA_SOURCE_COLUMN = "dataset_source"
COMPETITION_SOURCE_VALUE = "competition"
TELCO_SOURCE_VALUE = "telco_original"


def _sanitize_total_charges(frame: pd.DataFrame, *, fallback_median: float | None = None) -> pd.DataFrame:
    out = frame.copy()
    if "TotalCharges" not in out.columns:
        return out
    numeric = pd.to_numeric(out["TotalCharges"], errors="coerce")
    median = float(numeric.median()) if fallback_median is None else float(fallback_median)
    out["TotalCharges"] = numeric.fillna(median)
    return out


def _prepare_external_original_frame(path: str | Path) -> pd.DataFrame:
    original_path = Path(path)
    if not original_path.exists():
        raise FileNotFoundError(
            f"External original CSV not found: {original_path}. "
            "Provide --original-csv explicitly or place the file at the default path."
        )
    original_df = load_csv(original_path)
    if TARGET_COLUMN not in original_df.columns:
        raise ValueError(f"{original_path} must contain '{TARGET_COLUMN}'")
    if EXTERNAL_ID_COLUMN in original_df.columns and ID_COLUMN not in original_df.columns:
        original_df = original_df.rename(columns={EXTERNAL_ID_COLUMN: ID_COLUMN})
    fallback_median = float(pd.to_numeric(original_df["TotalCharges"], errors="coerce").median())
    original_df = _sanitize_total_charges(original_df, fallback_median=fallback_median)
    return original_df


def _build_overlap_signature(frame: pd.DataFrame, *, columns: Sequence[str]) -> pd.Series:
    normalized = frame.loc[:, list(columns)].copy()
    for column in normalized.columns:
        if column == "TotalCharges":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round(2).fillna(-999999.0)
        elif column == "MonthlyCharges":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round(2).fillna(-999999.0)
        else:
            normalized[column] = normalized[column].fillna("__nan__").astype(str)
    return pd.util.hash_pandas_object(normalized, index=False)


def _drop_exact_external_overlaps(
    original_df: pd.DataFrame,
    *,
    competition_train_df: pd.DataFrame,
    competition_test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    common_columns = sorted(
        (
            set(original_df.columns)
            & set(competition_train_df.columns)
            & set(competition_test_df.columns)
        )
        - {ID_COLUMN, TARGET_COLUMN}
    )
    if not common_columns:
        return original_df.copy(), 0

    competition_signatures = pd.concat(
        [
            _build_overlap_signature(competition_train_df, columns=common_columns),
            _build_overlap_signature(competition_test_df, columns=common_columns),
        ],
        axis=0,
        ignore_index=True,
    )
    overlap_mask = _build_overlap_signature(original_df, columns=common_columns).isin(set(competition_signatures.tolist()))
    filtered = original_df.loc[~overlap_mask].reset_index(drop=True)
    return filtered, int(overlap_mask.sum())


def _train_external_teacher(
    original_df: pd.DataFrame,
    *,
    teacher_params: CatBoostHyperParams,
    valid_size: float,
    early_stopping_rounds: int,
    verbose: int,
) -> tuple[Any, dict[str, Any]]:
    x_original, y_original = prepare_train_features(original_df, drop_id=True, feature_blocks=[])
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_original,
        y_original,
        test_size=float(valid_size),
        random_state=int(teacher_params.random_seed),
        stratify=y_original,
    )
    cat_columns = infer_categorical_columns(x_train)
    holdout_model = fit_with_validation(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        cat_columns=cat_columns,
        params=teacher_params,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
    holdout_pred = predict_proba(holdout_model, x_valid)
    final_iterations = int(best_iteration_or_default(holdout_model, teacher_params.iterations))

    full_params = CatBoostHyperParams(
        iterations=final_iterations,
        learning_rate=teacher_params.learning_rate,
        depth=teacher_params.depth,
        l2_leaf_reg=teacher_params.l2_leaf_reg,
        random_seed=teacher_params.random_seed,
        loss_function=teacher_params.loss_function,
        eval_metric=teacher_params.eval_metric,
    )
    full_model = fit_full_train(
        x_train=x_original,
        y_train=y_original,
        cat_columns=infer_categorical_columns(x_original),
        params=full_params,
        verbose=verbose,
    )
    metadata = {
        "rows": int(len(original_df)),
        "holdout_rows": int(len(x_valid)),
        "holdout_auc": float(binary_auc(y_valid, holdout_pred)),
        "best_iteration": int(holdout_model.get_best_iteration()),
        "final_iterations": int(final_iterations),
        "teacher_params": teacher_params.to_catboost_kwargs(),
    }
    return full_model, metadata


def _build_external_prediction_feature(
    model: Any,
    *,
    competition_train_df: pd.DataFrame,
    competition_test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = _sanitize_total_charges(
        competition_train_df.drop(columns=[TARGET_COLUMN], errors="ignore"),
        fallback_median=float(pd.to_numeric(competition_train_df["TotalCharges"], errors="coerce").median()),
    )
    test_features = _sanitize_total_charges(
        competition_test_df.copy(),
        fallback_median=float(pd.to_numeric(competition_train_df["TotalCharges"], errors="coerce").median()),
    )
    x_train_external = prepare_test_features(train_features, drop_id=True, feature_blocks=[])
    x_test_external = prepare_test_features(test_features, drop_id=True, feature_blocks=[])
    train_pred = predict_proba(model, x_train_external)
    test_pred = predict_proba(model, x_test_external)

    transfer_train = pd.DataFrame(
        {
            ID_COLUMN: competition_train_df[ID_COLUMN].astype("int64").values,
            TRANSFER_FEATURE_COLUMN: train_pred.astype(float).values,
        }
    )
    transfer_test = pd.DataFrame(
        {
            ID_COLUMN: competition_test_df[ID_COLUMN].astype("int64").values,
            TRANSFER_FEATURE_COLUMN: test_pred.astype(float).values,
        }
    )
    return transfer_train, transfer_test


def _prepare_joint_source_frames(
    competition_train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    *,
    stateless_blocks: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x_competition, y_competition = prepare_train_features(
        competition_train_df,
        drop_id=True,
        feature_blocks=stateless_blocks,
    )
    x_external, y_external = prepare_train_features(
        external_df,
        drop_id=True,
        feature_blocks=stateless_blocks,
    )
    x_competition = x_competition.copy()
    x_external = x_external.copy()
    x_competition[DATA_SOURCE_COLUMN] = COMPETITION_SOURCE_VALUE
    x_external[DATA_SOURCE_COLUMN] = TELCO_SOURCE_VALUE

    expected_columns = list(x_competition.columns)
    missing_external = sorted(set(expected_columns).difference(x_external.columns))
    extra_external = sorted(set(x_external.columns).difference(expected_columns))
    if missing_external or extra_external:
        raise ValueError(
            "Competition and external feature matrices are not column-compatible for joint training: "
            f"missing_external={missing_external}, extra_external={extra_external}"
        )
    x_external = x_external.reindex(columns=expected_columns)
    return x_competition, y_competition, x_external, y_external


def run_telco_joint_training_smoke(
    *,
    train_csv_path: str,
    test_csv_path: str,
    original_csv_path: str,
    challenger_params: CatBoostHyperParams,
    feature_blocks: Sequence[str],
    folds: int,
    seed: int,
    stratify_mode: str,
    external_weight: float,
    challenger_early_stopping_rounds: int,
    verbose: int,
) -> dict[str, Any]:
    """Run source-aware joint training with external Telco rows and OOF on competition rows only."""
    if float(external_weight) <= 0.0:
        raise ValueError("external_weight must be > 0.")

    competition_train_df = load_csv(train_csv_path)
    competition_test_df = load_csv(test_csv_path)
    original_df = _prepare_external_original_frame(original_csv_path)
    original_filtered_df, dropped_overlaps = _drop_exact_external_overlaps(
        original_df,
        competition_train_df=competition_train_df,
        competition_test_df=competition_test_df,
    )
    if original_filtered_df.empty:
        raise ValueError("External Telco dataset is empty after dropping exact overlaps with competition train/test.")

    normalized_blocks, stateless_blocks, stateful_blocks, _, _ = _prepare_train_matrix(
        competition_train_df,
        feature_blocks,
    )
    x_competition, y_competition, x_external, y_external = _prepare_joint_source_frames(
        competition_train_df,
        original_filtered_df,
        stateless_blocks=stateless_blocks,
    )

    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    stratify_labels = _build_stratify_labels(
        x_competition,
        y_competition,
        stratify_mode=stratify_mode,
        min_count=int(folds),
    )
    oof_pred = pd.Series(index=x_competition.index, dtype="float64", name="oof_pred")
    fold_metrics: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x_competition, stratify_labels), start=1):
        x_train_comp = x_competition.iloc[train_idx]
        y_train_comp = y_competition.iloc[train_idx]
        x_valid = x_competition.iloc[valid_idx]
        y_valid = y_competition.iloc[valid_idx]

        x_train_joint = pd.concat([x_train_comp, x_external], axis=0, ignore_index=True)
        y_train_joint = pd.concat([y_train_comp, y_external], axis=0, ignore_index=True)
        sample_weight_joint = pd.Series(1.0, index=x_train_joint.index, dtype="float64")
        sample_weight_joint.iloc[len(x_train_comp) :] = float(external_weight)

        x_train_fit, x_valid_fit = _transform_pair_with_stateful_blocks(x_train_joint, x_valid, stateful_blocks)
        cat_columns = infer_categorical_columns(x_train_fit)
        fold_params = CatBoostHyperParams(
            iterations=challenger_params.iterations,
            learning_rate=challenger_params.learning_rate,
            depth=challenger_params.depth,
            l2_leaf_reg=challenger_params.l2_leaf_reg,
            random_seed=seed,
            loss_function=challenger_params.loss_function,
            eval_metric=challenger_params.eval_metric,
            monotone_constraints=challenger_params.monotone_constraints,
        )
        model = fit_with_validation(
            x_train=x_train_fit,
            y_train=y_train_joint,
            x_valid=x_valid_fit,
            y_valid=y_valid,
            cat_columns=cat_columns,
            params=fold_params,
            early_stopping_rounds=challenger_early_stopping_rounds,
            verbose=verbose,
            sample_weight_train=sample_weight_joint.loc[x_train_fit.index],
        )
        fold_pred = predict_proba(model, x_valid_fit)
        oof_pred.iloc[valid_idx] = fold_pred.values
        fold_metrics.append(
            {
                "fold": int(fold_number),
                "auc": float(binary_auc(y_valid, fold_pred)),
                "best_iteration": int(model.get_best_iteration()),
                "final_iterations": int(best_iteration_or_default(model, challenger_params.iterations)),
                "competition_train_rows": int(len(x_train_comp)),
                "external_train_rows": int(len(x_external)),
                "valid_rows": int(len(valid_idx)),
                "external_weight": float(external_weight),
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError("Telco joint-training OOF predictions contain missing values.")

    fold_aucs = [float(item["auc"]) for item in fold_metrics]
    return {
        "generated_at_utc": utc_now_iso(),
        "train_csv_path": str(train_csv_path),
        "test_csv_path": str(test_csv_path),
        "original_csv_path": str(original_csv_path),
        "input_files": {
            "train_csv": describe_file(train_csv_path),
            "test_csv": describe_file(test_csv_path),
            "original_csv": describe_file(original_csv_path),
        },
        "feature_blocks": list(normalized_blocks),
        "joint_source_column": DATA_SOURCE_COLUMN,
        "joint_source_values": {
            "competition": COMPETITION_SOURCE_VALUE,
            "external": TELCO_SOURCE_VALUE,
        },
        "external_weight": float(external_weight),
        "teacher_metadata": {
            "rows_original": int(len(original_df)),
            "rows_after_overlap_filter": int(len(original_filtered_df)),
            "dropped_exact_overlaps": int(dropped_overlaps),
        },
        "rows": int(len(competition_train_df)),
        "external_rows": int(len(original_filtered_df)),
        "folds": int(folds),
        "seed": int(seed),
        "cv_mean_auc": float(np.mean(fold_aucs)),
        "cv_std_auc": float(np.std(fold_aucs)),
        "oof_auc": float(binary_auc(y_competition, oof_pred)),
        "fold_metrics": fold_metrics,
        "oof_pred": oof_pred,
    }


def run_telco_transfer_smoke(
    *,
    train_csv_path: str,
    test_csv_path: str,
    original_csv_path: str,
    teacher_params: CatBoostHyperParams,
    challenger_params: CatBoostHyperParams,
    feature_blocks: Sequence[str],
    folds: int,
    seed: int,
    stratify_mode: str,
    teacher_valid_size: float,
    teacher_early_stopping_rounds: int,
    challenger_early_stopping_rounds: int,
    verbose: int,
) -> dict[str, Any]:
    """Run the minimal external-teacher transfer feature smoke experiment.

    Parameters define:
    - competition train/test CSV paths
    - external original Telco CSV path
    - CatBoost params for the external teacher and the competition challenger
    - smoke CV controls for the challenger

    Returns a metrics dict with:
    - input file metadata/hashes
    - teacher training metadata
    - challenger CV metrics and fold metrics
    - `oof_pred` for the competition train set
    - `transfer_train` / `transfer_test` frames with `id` and `external_telco_pred`
    """
    competition_train_df = load_csv(train_csv_path)
    competition_test_df = load_csv(test_csv_path)
    original_df = _prepare_external_original_frame(original_csv_path)
    original_filtered_df, dropped_overlaps = _drop_exact_external_overlaps(
        original_df,
        competition_train_df=competition_train_df,
        competition_test_df=competition_test_df,
    )
    teacher_model, teacher_metadata = _train_external_teacher(
        original_filtered_df,
        teacher_params=teacher_params,
        valid_size=teacher_valid_size,
        early_stopping_rounds=teacher_early_stopping_rounds,
        verbose=verbose,
    )
    transfer_train, transfer_test = _build_external_prediction_feature(
        teacher_model,
        competition_train_df=competition_train_df,
        competition_test_df=competition_test_df,
    )

    normalized_blocks, _, stateful_blocks, x, y = _prepare_train_matrix(competition_train_df, feature_blocks)
    x = x.copy()
    x[TRANSFER_FEATURE_COLUMN] = transfer_train[TRANSFER_FEATURE_COLUMN].astype(float).values
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    stratify_labels = _build_stratify_labels(x, y, stratify_mode=stratify_mode, min_count=int(folds))

    oof_pred = pd.Series(index=x.index, dtype="float64", name="oof_pred")
    fold_metrics: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, stratify_labels), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        x_train_fit, x_valid_fit = _transform_pair_with_stateful_blocks(x_train, x_valid, stateful_blocks)
        cat_columns = infer_categorical_columns(x_train_fit)
        fold_params = CatBoostHyperParams(
            iterations=challenger_params.iterations,
            learning_rate=challenger_params.learning_rate,
            depth=challenger_params.depth,
            l2_leaf_reg=challenger_params.l2_leaf_reg,
            random_seed=seed,
            loss_function=challenger_params.loss_function,
            eval_metric=challenger_params.eval_metric,
            monotone_constraints=challenger_params.monotone_constraints,
        )
        model = fit_with_validation(
            x_train=x_train_fit,
            y_train=y_train,
            x_valid=x_valid_fit,
            y_valid=y_valid,
            cat_columns=cat_columns,
            params=fold_params,
            early_stopping_rounds=challenger_early_stopping_rounds,
            verbose=verbose,
        )
        fold_pred = predict_proba(model, x_valid_fit)
        oof_pred.iloc[valid_idx] = fold_pred.values
        fold_metrics.append(
            {
                "fold": int(fold_number),
                "auc": float(binary_auc(y_valid, fold_pred)),
                "best_iteration": int(model.get_best_iteration()),
                "final_iterations": int(best_iteration_or_default(model, challenger_params.iterations)),
                "valid_rows": int(len(valid_idx)),
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError("Telco transfer OOF predictions contain missing values.")

    fold_aucs = [float(item["auc"]) for item in fold_metrics]
    return {
        "generated_at_utc": utc_now_iso(),
        "train_csv_path": str(train_csv_path),
        "test_csv_path": str(test_csv_path),
        "original_csv_path": str(original_csv_path),
        "input_files": {
            "train_csv": describe_file(train_csv_path),
            "test_csv": describe_file(test_csv_path),
            "original_csv": describe_file(original_csv_path),
        },
        "feature_blocks": list(normalize_feature_blocks(feature_blocks)),
        "transfer_feature_column": TRANSFER_FEATURE_COLUMN,
        "teacher_metadata": {
            **teacher_metadata,
            "dropped_exact_overlaps": int(dropped_overlaps),
        },
        "rows": int(len(competition_train_df)),
        "folds": int(folds),
        "seed": int(seed),
        "cv_mean_auc": float(np.mean(fold_aucs)),
        "cv_std_auc": float(np.std(fold_aucs)),
        "oof_auc": float(binary_auc(y, oof_pred)),
        "fold_metrics": fold_metrics,
        "oof_pred": oof_pred,
        "transfer_train": transfer_train,
        "transfer_test": transfer_test,
    }
