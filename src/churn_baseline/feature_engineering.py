"""Feature engineering blocks for churn experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd


BLOCK_A: Final[str] = "A"
BLOCK_B: Final[str] = "B"
BLOCK_C: Final[str] = "C"
BLOCK_O: Final[str] = "O"
BLOCK_P: Final[str] = "P"
SUPPORTED_BLOCKS: Final[tuple[str, ...]] = (BLOCK_A, BLOCK_B, BLOCK_C, BLOCK_O, BLOCK_P)

# p01/p99 thresholds estimated from train.csv during outlier audit.
OUTLIER_FLAG_BOUNDS_P01_P99: Final[dict[str, tuple[float, float]]] = {
    "MonthlyCharges": (19.3, 115.5),
    "TotalCharges": (33.9, 8240.85),
    "monthly_per_tenure": (0.278472, 80.15),
    "total_per_tenure": (17.983333, 116.734722),
    "total_minus_monthly_tenure": (-911.064, 789.9035),
}
BLOCK_ALIASES: Final[dict[str, str]] = {
    "OUTLIER": BLOCK_O,
    "OUTLIERS": BLOCK_O,
    "OUTLIER_FLAGS": BLOCK_O,
    "FLAGS": BLOCK_O,
    "PCLIP": BLOCK_P,
    "CLIP": BLOCK_P,
    "CLIPPING": BLOCK_P,
    "PERCENTILE_CLIP": BLOCK_P,
}


def normalize_feature_blocks(feature_blocks: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize and validate requested feature blocks."""
    if not feature_blocks:
        return ()

    normalized = []
    seen = set()
    for value in feature_blocks:
        block = str(value).strip().upper()
        block = BLOCK_ALIASES.get(block, block)
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
        if block == BLOCK_O:
            out = _apply_block_o(out)
            continue
        if block == BLOCK_P:
            out = _apply_block_p(out)
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


def _apply_block_o(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("tenure", "MonthlyCharges", "TotalCharges"), block=BLOCK_O)
    out = frame.copy()

    numeric = _build_outlier_numeric_features(out)
    outlier_flag_columns = _apply_bounds(
        out=out,
        numeric=numeric,
        bounds=OUTLIER_FLAG_BOUNDS_P01_P99,
        prefix="is_outlier_",
        clip=False,
    )

    out["outlier_flag_count"] = out[outlier_flag_columns].sum(axis=1).astype("int8")
    return out


def _apply_block_p(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("tenure", "MonthlyCharges", "TotalCharges"), block=BLOCK_P)
    out = frame.copy()

    numeric = _build_outlier_numeric_features(out)
    clipped_columns = _apply_bounds(
        out=out,
        numeric=numeric,
        bounds=OUTLIER_FLAG_BOUNDS_P01_P99,
        prefix="pclip_",
        clip=True,
    )

    clipped_values = out[clipped_columns]
    out["pclip_sum"] = clipped_values.sum(axis=1)
    return out


def _build_outlier_numeric_features(frame: pd.DataFrame) -> dict[str, pd.Series]:
    tenure_values = pd.to_numeric(frame["tenure"], errors="coerce")
    tenure_safe = tenure_values.clip(lower=1).fillna(1.0)
    monthly_charges = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").fillna(0.0)
    total_charges = pd.to_numeric(frame["TotalCharges"], errors="coerce").fillna(0.0)

    return {
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "monthly_per_tenure": monthly_charges / tenure_safe,
        "total_per_tenure": total_charges / tenure_safe,
        "total_minus_monthly_tenure": total_charges - (monthly_charges * tenure_safe),
    }


def _apply_bounds(
    out: pd.DataFrame,
    numeric: dict[str, pd.Series],
    bounds: dict[str, tuple[float, float]],
    prefix: str,
    clip: bool,
) -> list[str]:
    missing_features = [feature_name for feature_name in bounds if feature_name not in numeric]
    if missing_features:
        raise ValueError(
            "Outlier bounds reference features missing from computed numeric set: "
            f"{missing_features}"
        )

    columns = []
    for feature_name, (lower_bound, upper_bound) in bounds.items():
        values = numeric[feature_name]
        column_name = f"{prefix}{feature_name}"
        if clip:
            out[column_name] = values.clip(lower=lower_bound, upper=upper_bound)
        else:
            out[column_name] = ((values < lower_bound) | (values > upper_bound)).astype("int8")
        columns.append(column_name)
    return columns
