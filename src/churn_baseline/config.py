"""Configuration primitives for the churn baseline."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


TARGET_COLUMN = "Churn"
ID_COLUMN = "id"
POSITIVE_LABEL = "Yes"


@dataclass(frozen=True)
class CatBoostHyperParams:
    """Hyperparameters for CatBoost binary classification."""

    iterations: int = 2200
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 5.0
    random_seed: int = 42
    loss_function: str = "Logloss"
    eval_metric: str = "AUC"

    def to_catboost_kwargs(self) -> Dict[str, Any]:
        """Return kwargs accepted by CatBoostClassifier."""
        return asdict(self)
