"""Model training and inference helpers."""

from collections.abc import Sequence
from pathlib import Path
from typing import List

import pandas as pd
from catboost import CatBoostClassifier, Pool

from .config import CatBoostHyperParams


def _resolve_monotone_constraints(
    params: CatBoostHyperParams,
    feature_columns: pd.Index | None,
) -> dict[int, int] | None:
    constraints = params.monotone_constraints
    if not constraints:
        return None
    if feature_columns is None:
        raise ValueError("feature_columns are required to resolve monotone_constraints.")

    missing_columns = [name for name in constraints if name not in feature_columns]
    if missing_columns:
        raise ValueError(
            "Monotone constraints reference features not present in the training matrix: "
            f"{missing_columns}"
        )

    return {int(feature_columns.get_loc(name)): int(direction) for name, direction in constraints.items()}


def build_model(
    params: CatBoostHyperParams,
    *,
    feature_columns: pd.Index | None = None,
) -> CatBoostClassifier:
    """Create a CatBoost model from hyperparameters."""
    kwargs = params.to_catboost_kwargs()
    if params.monotone_constraints:
        kwargs["monotone_constraints"] = _resolve_monotone_constraints(params, feature_columns)
    return CatBoostClassifier(**kwargs)


def fit_with_validation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_columns: List[str],
    params: CatBoostHyperParams,
    early_stopping_rounds: int,
    verbose: int,
    sample_weight_train: Sequence[float] | pd.Series | None = None,
    sample_weight_valid: Sequence[float] | pd.Series | None = None,
) -> CatBoostClassifier:
    """Train CatBoost with validation and early stopping."""
    model = build_model(params, feature_columns=x_train.columns)
    eval_set: tuple[pd.DataFrame, pd.Series] | Pool
    if sample_weight_valid is None:
        eval_set = (x_valid, y_valid)
    else:
        eval_set = Pool(
            data=x_valid,
            label=y_valid,
            cat_features=cat_columns,
            weight=sample_weight_valid,
        )
    model.fit(
        x_train,
        y_train,
        cat_features=cat_columns,
        sample_weight=sample_weight_train,
        eval_set=eval_set,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
    return model


def fit_full_train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cat_columns: List[str],
    params: CatBoostHyperParams,
    verbose: int,
    sample_weight: Sequence[float] | pd.Series | None = None,
) -> CatBoostClassifier:
    """Train CatBoost on the full train split without eval set."""
    model = build_model(params, feature_columns=x_train.columns)
    model.fit(
        x_train,
        y_train,
        cat_features=cat_columns,
        sample_weight=sample_weight,
        verbose=verbose,
    )
    return model


def predict_proba(model: CatBoostClassifier, features: pd.DataFrame) -> pd.Series:
    """Predict churn probability for class 1."""
    probs = model.predict_proba(features)[:, 1]
    return pd.Series(probs, index=features.index, name="Churn")


def save_model(model: CatBoostClassifier, path: str | Path) -> Path:
    """Persist CatBoost model to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    return out_path


def load_model(path: str | Path) -> CatBoostClassifier:
    """Load CatBoost model from disk."""
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def best_iteration_or_default(model: CatBoostClassifier, default_iterations: int) -> int:
    """Return best iteration if present, otherwise default iterations."""
    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter < 0:
        return default_iterations
    return int(best_iter) + 1
