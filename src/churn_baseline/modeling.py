"""Model training and inference helpers."""

from pathlib import Path
from typing import List, Optional

import pandas as pd
from catboost import CatBoostClassifier

from .config import CatBoostHyperParams


def build_model(params: CatBoostHyperParams) -> CatBoostClassifier:
    """Create a CatBoost model from hyperparameters."""
    return CatBoostClassifier(**params.to_catboost_kwargs())


def fit_with_validation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_columns: List[str],
    params: CatBoostHyperParams,
    early_stopping_rounds: int,
    verbose: int,
    sample_weight: Optional[pd.Series] = None,
) -> CatBoostClassifier:
    """Train CatBoost with validation and early stopping."""
    model = build_model(params)
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        cat_features=cat_columns,
        eval_set=(x_valid, y_valid),
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
    sample_weight: Optional[pd.Series] = None,
) -> CatBoostClassifier:
    """Train CatBoost on the full train split without eval set."""
    model = build_model(params)
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        cat_features=cat_columns,
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
