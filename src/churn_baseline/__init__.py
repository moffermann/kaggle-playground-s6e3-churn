"""Baseline package for Kaggle Playground S6E3 churn competition."""

from .config import CatBoostHyperParams
from .pipeline import (
    make_submission,
    make_submission_ensemble,
    train_baseline,
    train_baseline_cv,
    train_baseline_cv_multiseed,
)

__all__ = [
    "CatBoostHyperParams",
    "make_submission",
    "make_submission_ensemble",
    "train_baseline",
    "train_baseline_cv",
    "train_baseline_cv_multiseed",
]
