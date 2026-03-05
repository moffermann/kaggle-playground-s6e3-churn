"""Data loading and feature/target preparation."""

from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

from .config import ID_COLUMN, POSITIVE_LABEL, TARGET_COLUMN
from .feature_engineering import apply_feature_engineering


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a dataframe."""
    return pd.read_csv(path)


def encode_target(series: pd.Series) -> pd.Series:
    """Map Yes/No target labels to 1/0."""
    return (series.astype(str).str.lower() == POSITIVE_LABEL.lower()).astype("int8")


def prepare_train_features(
    train_df: pd.DataFrame,
    drop_id: bool = True,
    feature_blocks: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split train dataframe into features and encoded target."""
    x = train_df.drop(columns=[TARGET_COLUMN]).copy()
    if drop_id and ID_COLUMN in x.columns:
        x = x.drop(columns=[ID_COLUMN])
    x = apply_feature_engineering(x, feature_blocks)
    y = encode_target(train_df[TARGET_COLUMN])
    return x, y


def prepare_test_features(
    test_df: pd.DataFrame,
    drop_id: bool = True,
    feature_blocks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Prepare test feature dataframe using the same feature policy as train."""
    x = test_df.copy()
    if drop_id and ID_COLUMN in x.columns:
        x = x.drop(columns=[ID_COLUMN])
    x = apply_feature_engineering(x, feature_blocks)
    return x


def infer_categorical_columns(frame: pd.DataFrame) -> List[str]:
    """Infer categorical columns from object dtype."""
    return frame.select_dtypes(include=["object", "category"]).columns.tolist()
