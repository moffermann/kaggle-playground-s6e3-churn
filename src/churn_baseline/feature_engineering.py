"""Feature engineering blocks for churn experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd


BLOCK_A: Final[str] = "A"
BLOCK_B: Final[str] = "B"
BLOCK_C: Final[str] = "C"
SUPPORTED_BLOCKS: Final[tuple[str, ...]] = (BLOCK_A, BLOCK_B, BLOCK_C)


def normalize_feature_blocks(feature_blocks: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize and validate requested feature blocks."""
    if not feature_blocks:
        return ()

    normalized = []
    seen = set()
    for value in feature_blocks:
        block = str(value).strip().upper()
        if not block:
            continue
        if block not in SUPPORTED_BLOCKS:
            raise ValueError(f"Unsupported feature block '{value}'. Supported: {SUPPORTED_BLOCKS}")
        if block not in seen:
            seen.add(block)
            normalized.append(block)
    return tuple(normalized)


def apply_feature_engineering(
    features: pd.DataFrame,
    feature_blocks: Sequence[str] | None,
) -> pd.DataFrame:
    """Apply selected feature engineering blocks."""
    out = features.copy()
    blocks = normalize_feature_blocks(feature_blocks)
    for block in blocks:
        if block == BLOCK_A:
            out = _apply_block_a(out)
            continue
        if block == BLOCK_B:
            out = _apply_block_b(out)
            continue
        if block == BLOCK_C:
            out = _apply_block_c(out)
            continue
    return out


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], block: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Feature block {block} requires columns not present in dataframe: {missing}")


def _apply_block_a(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("tenure", "MonthlyCharges", "Contract"), block=BLOCK_A)
    out = frame.copy()

    tenure_values = pd.to_numeric(out["tenure"], errors="coerce")
    tenure_labels = ["0_6", "7_12", "13_24", "25_48", "49_plus"]
    out["tenure_bin"] = pd.cut(
        tenure_values,
        bins=[-np.inf, 6, 12, 24, 48, np.inf],
        labels=tenure_labels,
    ).astype(str)

    out["is_new_customer"] = (tenure_values <= 6).fillna(False).astype("int8")

    tenure_safe = tenure_values.clip(lower=1).fillna(1.0)
    monthly_charges = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0.0)
    out["monthly_per_tenure"] = monthly_charges / tenure_safe

    out["tenure_x_contract"] = out["tenure_bin"].astype(str) + "__" + out["Contract"].astype(str)
    return out


def _apply_block_b(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("PaymentMethod", "Contract", "InternetService"), block=BLOCK_B)
    out = frame.copy()

    payment_method = out["PaymentMethod"].astype(str)
    payment_method_norm = payment_method.str.lower()
    out["is_auto_payment"] = payment_method_norm.str.contains("automatic", regex=False).astype("int8")
    out["is_manual_payment"] = (1 - out["is_auto_payment"]).astype("int8")
    out["is_electronic_check"] = payment_method_norm.eq("electronic check").astype("int8")

    out["payment_x_contract"] = payment_method + "__" + out["Contract"].astype(str)
    out["payment_x_internet"] = payment_method + "__" + out["InternetService"].astype(str)
    return out


def _apply_block_c(frame: pd.DataFrame) -> pd.DataFrame:
    service_columns = (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    _require_columns(frame, service_columns + ("InternetService", "PhoneService"), block=BLOCK_C)
    out = frame.copy()

    addons_active = np.zeros(len(out), dtype=np.int16)
    for column in service_columns:
        addons_active += out[column].astype(str).str.lower().eq("yes").to_numpy(dtype=np.int16)
    out["addons_active_count"] = addons_active.astype("int16")
    out["is_bundle_heavy"] = (out["addons_active_count"] >= 3).astype("int8")

    out["has_internet"] = out["InternetService"].astype(str).str.lower().ne("no").astype("int8")
    out["has_phone"] = out["PhoneService"].astype(str).str.lower().eq("yes").astype("int8")
    out["techsupport_x_onlinesecurity"] = (
        out["TechSupport"].astype(str) + "__" + out["OnlineSecurity"].astype(str)
    )
    return out
