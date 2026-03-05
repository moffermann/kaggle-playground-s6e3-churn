"""Evaluation utilities."""

from sklearn.metrics import roc_auc_score


def binary_auc(y_true, y_score) -> float:
    """Compute ROC AUC for binary classification."""
    return float(roc_auc_score(y_true, y_score))
