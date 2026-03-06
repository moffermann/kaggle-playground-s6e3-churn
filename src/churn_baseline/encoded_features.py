"""Feature encoding helpers for non-CatBoost tree boosters."""

from __future__ import annotations

import pandas as pd


def build_dense_encoded_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed-type tabular frame to dense numeric matrix via one-hot encoding."""
    encoded = pd.get_dummies(features, drop_first=False, dtype="float32")
    encoded = encoded.reindex(sorted(encoded.columns), axis=1)
    return encoded.astype("float32")
