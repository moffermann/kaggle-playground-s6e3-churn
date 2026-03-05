"""High-level baseline training and submission pipeline."""

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .artifacts import write_json
from .config import CatBoostHyperParams, ID_COLUMN, TARGET_COLUMN
from .data import (
    infer_categorical_columns,
    load_csv,
    prepare_test_features,
    prepare_train_features,
)
from .evaluation import binary_auc
from .modeling import (
    best_iteration_or_default,
    fit_full_train,
    fit_with_validation,
    load_model,
    predict_proba,
    save_model,
)


def train_baseline(
    train_csv_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    params: CatBoostHyperParams,
    valid_size: float,
    random_state: int,
    early_stopping_rounds: int,
    verbose: int,
) -> Dict[str, Any]:
    """Train holdout+full baseline and save model + metrics."""
    train_df = load_csv(train_csv_path)
    x, y = prepare_train_features(train_df, drop_id=True)
    cat_columns = infer_categorical_columns(x)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=valid_size,
        random_state=random_state,
        stratify=y,
    )

    holdout_model = fit_with_validation(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        cat_columns=cat_columns,
        params=params,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )

    valid_pred = predict_proba(holdout_model, x_valid)
    holdout_auc = binary_auc(y_valid, valid_pred)
    final_iterations = best_iteration_or_default(holdout_model, params.iterations)

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
        x_train=x,
        y_train=y,
        cat_columns=cat_columns,
        params=full_params,
        verbose=verbose,
    )
    save_model(full_model, model_path)

    metrics: Dict[str, Any] = {
        "train_rows": int(len(train_df)),
        "feature_count": int(x.shape[1]),
        "categorical_columns": cat_columns,
        "holdout_auc": holdout_auc,
        "holdout_best_iteration": int(holdout_model.get_best_iteration()),
        "final_iterations": int(final_iterations),
        "params_holdout": params.to_catboost_kwargs(),
        "params_full_train": full_params.to_catboost_kwargs(),
        "model_path": str(model_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
    }
    write_json(metrics_path, metrics)
    return metrics


def _run_cv_for_seed(
    x: pd.DataFrame,
    y: pd.Series,
    cat_columns: list[str],
    params: CatBoostHyperParams,
    folds: int,
    seed: int,
    early_stopping_rounds: int,
    verbose: int,
) -> Dict[str, Any]:
    """Run Stratified K-Fold CV for a single random seed and return OOF stats."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_pred = pd.Series(index=x.index, dtype="float64", name=f"oof_seed_{seed}")
    fold_rows = []
    fold_iterations = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        fold_params = CatBoostHyperParams(
            iterations=params.iterations,
            learning_rate=params.learning_rate,
            depth=params.depth,
            l2_leaf_reg=params.l2_leaf_reg,
            random_seed=seed,
            loss_function=params.loss_function,
            eval_metric=params.eval_metric,
        )

        fold_model = fit_with_validation(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            cat_columns=cat_columns,
            params=fold_params,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        fold_pred = predict_proba(fold_model, x_valid)
        oof_pred.iloc[valid_idx] = fold_pred.values
        fold_auc = binary_auc(y_valid, fold_pred)
        fold_final_iterations = best_iteration_or_default(fold_model, params.iterations)
        fold_iterations.append(fold_final_iterations)
        fold_rows.append(
            {
                "fold": fold_number,
                "valid_rows": int(len(valid_idx)),
                "auc": fold_auc,
                "best_iteration": int(fold_model.get_best_iteration()),
                "final_iterations": int(fold_final_iterations),
                "seed": int(seed),
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError(f"OOF predictions contain missing values for seed {seed}")

    fold_aucs = [row["auc"] for row in fold_rows]
    final_iterations = int(round(float(np.mean(fold_iterations))))
    final_iterations = max(final_iterations, 1)

    return {
        "seed": int(seed),
        "oof_pred": oof_pred,
        "fold_metrics": fold_rows,
        "fold_final_iterations": [int(v) for v in fold_iterations],
        "cv_mean_auc": float(np.mean(fold_aucs)),
        "cv_std_auc": float(np.std(fold_aucs)),
        "oof_auc": binary_auc(y, oof_pred),
        "final_iterations": int(final_iterations),
    }


def train_baseline_cv(
    train_csv_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
    params: CatBoostHyperParams,
    folds: int,
    random_state: int,
    early_stopping_rounds: int,
    verbose: int,
) -> Dict[str, Any]:
    """Train baseline with Stratified K-Fold CV, generate OOF and fit final model."""
    if folds < 2:
        raise ValueError("folds must be >= 2")

    train_df = load_csv(train_csv_path)
    x, y = prepare_train_features(train_df, drop_id=True)
    cat_columns = infer_categorical_columns(x)

    seed_run = _run_cv_for_seed(
        x=x,
        y=y,
        cat_columns=cat_columns,
        params=params,
        folds=folds,
        seed=random_state,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
    oof_pred = seed_run["oof_pred"]
    fold_rows = seed_run["fold_metrics"]
    fold_iterations = seed_run["fold_final_iterations"]
    cv_mean_auc = seed_run["cv_mean_auc"]
    cv_std_auc = seed_run["cv_std_auc"]
    oof_auc = seed_run["oof_auc"]
    final_iterations = seed_run["final_iterations"]

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
        x_train=x,
        y_train=y,
        cat_columns=cat_columns,
        params=full_params,
        verbose=verbose,
    )
    save_model(full_model, model_path)

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN],
            "target": y.astype(int).values,
            "oof_pred": oof_pred.values,
        }
    )
    oof_out_path = Path(oof_path)
    oof_out_path.parent.mkdir(parents=True, exist_ok=True)
    oof_output.to_csv(oof_out_path, index=False)

    metrics: Dict[str, Any] = {
        "train_rows": int(len(train_df)),
        "feature_count": int(x.shape[1]),
        "categorical_columns": cat_columns,
        "cv_folds": int(folds),
        "cv_fold_metrics": fold_rows,
        "cv_mean_auc": cv_mean_auc,
        "cv_std_auc": cv_std_auc,
        "oof_auc": oof_auc,
        "fold_final_iterations": [int(v) for v in fold_iterations],
        "final_iterations": int(final_iterations),
        "params_cv": params.to_catboost_kwargs(),
        "params_full_train": full_params.to_catboost_kwargs(),
        "model_path": str(model_path),
        "oof_path": str(oof_out_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
    }
    write_json(metrics_path, metrics)
    return metrics


def train_baseline_cv_multiseed(
    train_csv_path: str | Path,
    models_dir: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
    params: CatBoostHyperParams,
    folds: int,
    seeds: Sequence[int],
    early_stopping_rounds: int,
    verbose: int,
) -> Dict[str, Any]:
    """Train CV baseline across multiple seeds and average OOF predictions."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if not seeds:
        raise ValueError("seeds must not be empty")

    normalized_seeds = [int(seed) for seed in seeds]
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds contain duplicates")

    train_df = load_csv(train_csv_path)
    x, y = prepare_train_features(train_df, drop_id=True)
    cat_columns = infer_categorical_columns(x)

    models_out_dir = Path(models_dir)
    models_out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_metrics = []
    per_seed_oof = []
    model_paths = []

    for seed in normalized_seeds:
        seed_run = _run_cv_for_seed(
            x=x,
            y=y,
            cat_columns=cat_columns,
            params=params,
            folds=folds,
            seed=seed,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        per_seed_oof.append(seed_run["oof_pred"].values)

        full_params = CatBoostHyperParams(
            iterations=seed_run["final_iterations"],
            learning_rate=params.learning_rate,
            depth=params.depth,
            l2_leaf_reg=params.l2_leaf_reg,
            random_seed=seed,
            loss_function=params.loss_function,
            eval_metric=params.eval_metric,
        )

        model_path = models_out_dir / f"catboost_seed_{seed}.cbm"
        full_model = fit_full_train(
            x_train=x,
            y_train=y,
            cat_columns=cat_columns,
            params=full_params,
            verbose=verbose,
        )
        save_model(full_model, model_path)
        model_paths.append(str(model_path))

        per_seed_metrics.append(
            {
                "seed": seed_run["seed"],
                "cv_mean_auc": seed_run["cv_mean_auc"],
                "cv_std_auc": seed_run["cv_std_auc"],
                "oof_auc": seed_run["oof_auc"],
                "final_iterations": seed_run["final_iterations"],
                "fold_final_iterations": seed_run["fold_final_iterations"],
                "fold_metrics": seed_run["fold_metrics"],
                "model_path": str(model_path),
                "params_full_train": full_params.to_catboost_kwargs(),
            }
        )

    oof_matrix = np.column_stack(per_seed_oof)
    ensemble_oof = np.mean(oof_matrix, axis=1)
    ensemble_oof_auc = binary_auc(y, ensemble_oof)

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN],
            "target": y.astype(int).values,
        }
    )
    for idx, seed in enumerate(normalized_seeds):
        oof_output[f"oof_seed_{seed}"] = oof_matrix[:, idx]
    oof_output["oof_ensemble"] = ensemble_oof

    oof_out_path = Path(oof_path)
    oof_out_path.parent.mkdir(parents=True, exist_ok=True)
    oof_output.to_csv(oof_out_path, index=False)

    seed_cv_means = [item["cv_mean_auc"] for item in per_seed_metrics]
    seed_oof_aucs = [item["oof_auc"] for item in per_seed_metrics]
    metrics: Dict[str, Any] = {
        "train_rows": int(len(train_df)),
        "feature_count": int(x.shape[1]),
        "categorical_columns": cat_columns,
        "cv_folds": int(folds),
        "seeds": normalized_seeds,
        "per_seed_metrics": per_seed_metrics,
        "ensemble_oof_auc": float(ensemble_oof_auc),
        "mean_seed_cv_auc": float(np.mean(seed_cv_means)),
        "std_seed_cv_auc": float(np.std(seed_cv_means)),
        "mean_seed_oof_auc": float(np.mean(seed_oof_aucs)),
        "std_seed_oof_auc": float(np.std(seed_oof_aucs)),
        "model_paths": model_paths,
        "models_dir": str(models_out_dir),
        "oof_path": str(oof_out_path),
        "params_cv": params.to_catboost_kwargs(),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
    }
    write_json(metrics_path, metrics)
    return metrics


def make_submission(
    model_path: str | Path,
    test_csv_path: str | Path,
    output_csv_path: str | Path,
) -> Dict[str, Any]:
    """Generate submission CSV using a trained model."""
    test_df = load_csv(test_csv_path)
    if ID_COLUMN not in test_df.columns:
        raise ValueError(f"Column '{ID_COLUMN}' not found in test CSV")

    ids = test_df[ID_COLUMN].copy()
    x_test = prepare_test_features(test_df, drop_id=True)
    model = load_model(model_path)
    predictions = predict_proba(model, x_test)

    submission = pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: predictions})
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    return {
        "output_csv": str(out_path),
        "rows": int(len(submission)),
        "prediction_min": float(submission[TARGET_COLUMN].min()),
        "prediction_max": float(submission[TARGET_COLUMN].max()),
    }


def make_submission_ensemble(
    model_paths: Sequence[str | Path],
    test_csv_path: str | Path,
    output_csv_path: str | Path,
) -> Dict[str, Any]:
    """Generate submission CSV by averaging predictions from multiple models."""
    if not model_paths:
        raise ValueError("model_paths must not be empty")

    test_df = load_csv(test_csv_path)
    if ID_COLUMN not in test_df.columns:
        raise ValueError(f"Column '{ID_COLUMN}' not found in test CSV")

    ids = test_df[ID_COLUMN].copy()
    x_test = prepare_test_features(test_df, drop_id=True)

    predictions_matrix = []
    for model_path in model_paths:
        model = load_model(model_path)
        pred = predict_proba(model, x_test).values
        predictions_matrix.append(pred)
    mean_pred = np.mean(np.column_stack(predictions_matrix), axis=1)

    submission = pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: mean_pred})
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    return {
        "output_csv": str(out_path),
        "rows": int(len(submission)),
        "model_count": int(len(model_paths)),
        "prediction_min": float(submission[TARGET_COLUMN].min()),
        "prediction_max": float(submission[TARGET_COLUMN].max()),
    }
