"""Feature engineering blocks for churn experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


BLOCK_A: Final[str] = "A"
BLOCK_B: Final[str] = "B"
BLOCK_C: Final[str] = "C"
BLOCK_G: Final[str] = "G"
BLOCK_H: Final[str] = "H"
BLOCK_R: Final[str] = "R"
BLOCK_S: Final[str] = "S"
BLOCK_V: Final[str] = "V"
BLOCK_O: Final[str] = "O"
BLOCK_P: Final[str] = "P"
SUPPORTED_BLOCKS: Final[tuple[str, ...]] = (
    BLOCK_A,
    BLOCK_B,
    BLOCK_C,
    BLOCK_G,
    BLOCK_H,
    BLOCK_R,
    BLOCK_S,
    BLOCK_V,
    BLOCK_O,
    BLOCK_P,
)

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
    "RENEWAL": BLOCK_R,
    "LIFECYCLE": BLOCK_R,
    "CYCLE": BLOCK_R,
    "COVERAGE": BLOCK_G,
    "BACKOFF": BLOCK_G,
    "SUPPORT": BLOCK_G,
    "FAMILY": BLOCK_G,
    "HARD": BLOCK_H,
    "CONTRAST": BLOCK_H,
    "COHORT": BLOCK_H,
    "COHORT_CONTRAST": BLOCK_H,
    "SERVICE": BLOCK_S,
    "SERVICES": BLOCK_S,
    "BUNDLE": BLOCK_S,
    "TOPOLOGY": BLOCK_S,
    "VALUE": BLOCK_V,
    "PRESSURE": BLOCK_V,
    "FRICTION": BLOCK_V,
}
STATEFUL_BLOCKS: Final[tuple[str, ...]] = (BLOCK_G,)


@dataclass(frozen=True)
class CoverageBackoffState:
    """Train-fitted support maps for family coverage/backoff features."""

    min_segment5_support: int
    segment3_counts: dict[str, int]
    segment5_counts: dict[str, int]


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


def partition_feature_blocks(
    feature_blocks: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split requested blocks into stateless and fit-aware subsets."""
    normalized = normalize_feature_blocks(feature_blocks)
    stateless = tuple(block for block in normalized if block not in STATEFUL_BLOCKS)
    stateful = tuple(block for block in normalized if block in STATEFUL_BLOCKS)
    return stateless, stateful


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
        if block == BLOCK_G:
            raise ValueError(
                "Feature block G requires train-fitted coverage state; "
                "use pipeline helpers that fit on the training split first."
            )
        if block == BLOCK_H:
            out = _apply_block_h(out)
            continue
        if block == BLOCK_R:
            out = _apply_block_r(out)
            continue
        if block == BLOCK_S:
            out = _apply_block_s(out)
            continue
        if block == BLOCK_V:
            out = _apply_block_v(out)
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


def _build_tenure_bin(tenure_values: pd.Series) -> pd.Series:
    return pd.cut(
        tenure_values,
        bins=[-np.inf, 6, 12, 24, 48, np.inf],
        labels=["0_6", "7_12", "13_24", "25_48", "49_plus"],
    ).astype(str)


def _ensure_tenure_bin(frame: pd.DataFrame) -> pd.Series:
    if "tenure_bin" in frame.columns:
        return frame["tenure_bin"].astype(str)
    _require_columns(frame, ("tenure",), block=BLOCK_G)
    tenure_values = pd.to_numeric(frame["tenure"], errors="coerce")
    return _build_tenure_bin(tenure_values)


def _yes_no_flag(series: pd.Series, positive_value: str = "yes") -> pd.Series:
    return series.astype(str).str.lower().eq(positive_value).astype("int8")


def _group_size(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame.groupby(list(columns), dropna=False)[columns[0]].transform("size").astype("int32")


def _build_segment3_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["PaymentMethod"].astype(str)
        + "__"
        + frame["Contract"].astype(str)
        + "__"
        + frame["InternetService"].astype(str)
    )


def _build_segment5_key(frame: pd.DataFrame, tenure_bin: pd.Series) -> pd.Series:
    return (
        frame["PaymentMethod"].astype(str)
        + "__"
        + frame["Contract"].astype(str)
        + "__"
        + frame["InternetService"].astype(str)
        + "__"
        + frame["PaperlessBilling"].astype(str)
        + "__"
        + tenure_bin.astype(str)
    )


def _support_bucket(counts: pd.Series) -> pd.Series:
    labels = ["zero", "1_4", "5_24", "25_74", "75_199", "200p"]
    buckets = pd.cut(
        counts.astype("float64"),
        bins=[-0.1, 0.5, 4.5, 24.5, 74.5, 199.5, np.inf],
        labels=labels,
    )
    return buckets.astype(str)


def fit_coverage_backoff_state(
    frame: pd.DataFrame,
    *,
    min_segment5_support: int = 75,
) -> CoverageBackoffState:
    """Fit train-only support maps for detailed and parent families."""
    required = ("PaymentMethod", "Contract", "InternetService", "PaperlessBilling")
    _require_columns(frame, required + ("tenure",), block=BLOCK_G)

    tenure_bin = _ensure_tenure_bin(frame)
    segment3 = _build_segment3_key(frame)
    segment5 = _build_segment5_key(frame, tenure_bin)

    segment3_counts = segment3.value_counts(dropna=False).astype(int).to_dict()
    segment5_counts = segment5.value_counts(dropna=False).astype(int).to_dict()
    return CoverageBackoffState(
        min_segment5_support=int(min_segment5_support),
        segment3_counts=segment3_counts,
        segment5_counts=segment5_counts,
    )


def apply_coverage_backoff_features(
    frame: pd.DataFrame,
    state: CoverageBackoffState,
) -> pd.DataFrame:
    """Append train-fitted coverage and backoff features to any frame."""
    required = ("PaymentMethod", "Contract", "InternetService", "PaperlessBilling")
    _require_columns(frame, required + ("tenure",), block=BLOCK_G)

    out = frame.copy()
    tenure_bin = _ensure_tenure_bin(out)
    if "tenure_bin" not in out.columns:
        out["tenure_bin"] = tenure_bin

    segment3 = _build_segment3_key(out)
    segment5 = _build_segment5_key(out, tenure_bin)
    segment3_rows = segment3.map(state.segment3_counts).fillna(0).astype("int32")
    segment5_rows = segment5.map(state.segment5_counts).fillna(0).astype("int32")
    use_segment5 = segment5_rows >= int(state.min_segment5_support)
    segment3_rows_safe = segment3_rows.clip(lower=1)

    out["segment3_family"] = segment3.astype(str)
    out["segment5_family"] = segment5.astype(str)
    out["segment3_train_rows"] = segment3_rows
    out["segment5_train_rows"] = segment5_rows
    out["segment5_seen_in_fit"] = segment5_rows.gt(0).astype("int8")
    out["segment5_low_support_in_fit"] = segment5_rows.lt(int(state.min_segment5_support)).astype("int8")
    out["segment5_backoff_to_segment3"] = (~use_segment5).astype("int8")
    out["segment5_share_within_segment3"] = (
        segment5_rows.astype("float64") / segment3_rows_safe.astype("float64")
    ).astype("float64")
    out["segment3_support_bucket"] = _support_bucket(segment3_rows)
    out["segment5_support_bucket"] = _support_bucket(segment5_rows)
    out["segment5_backoff_family"] = np.where(
        use_segment5,
        segment5.astype(str),
        "BACKOFF__" + segment3.astype(str),
    )
    return out


def _apply_block_a(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("tenure", "MonthlyCharges", "Contract"), block=BLOCK_A)
    out = frame.copy()

    tenure_values = pd.to_numeric(out["tenure"], errors="coerce")
    out["tenure_bin"] = _build_tenure_bin(tenure_values)

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


def _apply_block_h(frame: pd.DataFrame) -> pd.DataFrame:
    service_columns = (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    required = (
        "PaymentMethod",
        "Contract",
        "InternetService",
        "PaperlessBilling",
        "tenure",
        "PhoneService",
        "MultipleLines",
    )
    _require_columns(frame, required + service_columns, block=BLOCK_H)
    out = frame.copy()

    tenure_values = pd.to_numeric(out["tenure"], errors="coerce").fillna(0.0)
    if "tenure_bin" not in out.columns:
        out["tenure_bin"] = _build_tenure_bin(tenure_values)

    has_internet = out["InternetService"].astype(str).str.lower().ne("no").astype("int8")
    has_phone = _yes_no_flag(out["PhoneService"])
    has_multiple_lines = _yes_no_flag(out["MultipleLines"])

    support_count = np.zeros(len(out), dtype=np.int16)
    entertainment_count = np.zeros(len(out), dtype=np.int16)
    for column in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"):
        support_count += _yes_no_flag(out[column]).to_numpy(dtype=np.int16)
    for column in ("StreamingTV", "StreamingMovies"):
        entertainment_count += _yes_no_flag(out[column]).to_numpy(dtype=np.int16)

    paid_services_count = (
        has_internet.to_numpy(dtype=np.int16)
        + has_phone.to_numpy(dtype=np.int16)
        + has_multiple_lines.to_numpy(dtype=np.int16)
        + support_count
        + entertainment_count
    ).astype("int16")
    addon_total = support_count.astype("float64") + entertainment_count.astype("float64")
    support_share = np.divide(
        support_count.astype("float64"),
        np.clip(addon_total, 1.0, None),
    )

    support_signature = (
        out["OnlineSecurity"].astype(str).str.slice(0, 1).str.upper()
        + out["OnlineBackup"].astype(str).str.slice(0, 1).str.upper()
        + out["DeviceProtection"].astype(str).str.slice(0, 1).str.upper()
        + out["TechSupport"].astype(str).str.slice(0, 1).str.upper()
    )
    streaming_signature = (
        out["StreamingTV"].astype(str).str.slice(0, 1).str.upper()
        + out["StreamingMovies"].astype(str).str.slice(0, 1).str.upper()
    )
    bundle_archetype = np.select(
        condlist=[
            has_internet.to_numpy(dtype=np.int8) == 0,
            (support_count == 0) & (entertainment_count == 0),
            (support_count == 0) & (entertainment_count >= 1),
            (support_count >= 1) & (entertainment_count == 0),
            support_count >= (entertainment_count + 2),
            entertainment_count >= (support_count + 1),
        ],
        choicelist=[
            "no_internet",
            "connectivity_only",
            "streaming_only",
            "support_only",
            "support_heavy",
            "entertainment_heavy",
        ],
        default="balanced",
    )

    detailed_group = ["PaymentMethod", "Contract", "InternetService", "PaperlessBilling", "tenure_bin"]
    fallback_group = ["Contract", "InternetService", "tenure_bin"]
    detailed_rows = _group_size(out, detailed_group)
    fallback_rows = _group_size(out, fallback_group)
    use_detailed = detailed_rows >= 75

    support_series = pd.Series(support_count, index=out.index, dtype="float64")
    entertainment_series = pd.Series(entertainment_count, index=out.index, dtype="float64")
    paid_services_series = pd.Series(paid_services_count, index=out.index, dtype="float64")
    support_share_series = pd.Series(support_share, index=out.index, dtype="float64")
    service_mix_series = pd.Series(support_count - entertainment_count, index=out.index, dtype="float64")

    detailed_support_median = support_series.groupby(
        [out[column] for column in detailed_group],
        dropna=False,
    ).transform("median")
    fallback_support_median = support_series.groupby(
        [out[column] for column in fallback_group],
        dropna=False,
    ).transform("median")
    detailed_entertainment_median = entertainment_series.groupby(
        [out[column] for column in detailed_group],
        dropna=False,
    ).transform("median")
    fallback_entertainment_median = entertainment_series.groupby(
        [out[column] for column in fallback_group],
        dropna=False,
    ).transform("median")
    detailed_paid_services_median = paid_services_series.groupby(
        [out[column] for column in detailed_group],
        dropna=False,
    ).transform("median")
    fallback_paid_services_median = paid_services_series.groupby(
        [out[column] for column in fallback_group],
        dropna=False,
    ).transform("median")
    detailed_support_share_mean = support_share_series.groupby(
        [out[column] for column in detailed_group],
        dropna=False,
    ).transform("mean")
    fallback_support_share_mean = support_share_series.groupby(
        [out[column] for column in fallback_group],
        dropna=False,
    ).transform("mean")
    detailed_service_mix_mean = service_mix_series.groupby(
        [out[column] for column in detailed_group],
        dropna=False,
    ).transform("mean")
    fallback_service_mix_mean = service_mix_series.groupby(
        [out[column] for column in fallback_group],
        dropna=False,
    ).transform("mean")

    detailed_support_rank = support_series.groupby(
        [out[column] for column in detailed_group],
        dropna=False,
    ).rank(method="average", pct=True)
    fallback_support_rank = support_series.groupby(
        [out[column] for column in fallback_group],
        dropna=False,
    ).rank(method="average", pct=True)
    detailed_entertainment_rank = entertainment_series.groupby(
        [out[column] for column in detailed_group],
        dropna=False,
    ).rank(method="average", pct=True)
    fallback_entertainment_rank = entertainment_series.groupby(
        [out[column] for column in fallback_group],
        dropna=False,
    ).rank(method="average", pct=True)
    detailed_service_mix_rank = service_mix_series.groupby(
        [out[column] for column in detailed_group],
        dropna=False,
    ).rank(method="average", pct=True)
    fallback_service_mix_rank = service_mix_series.groupby(
        [out[column] for column in fallback_group],
        dropna=False,
    ).rank(method="average", pct=True)

    bundle_key = pd.Series(bundle_archetype, index=out.index, dtype="object")
    service_signature_key = (
        out["InternetService"].astype(str)
        + "__"
        + support_signature.astype(str)
        + "__"
        + streaming_signature.astype(str)
        + "__"
        + out["MultipleLines"].astype(str)
    )
    detailed_bundle_rows = _group_size(out.assign(_contrast_bundle_key=bundle_key), detailed_group + ["_contrast_bundle_key"])
    fallback_bundle_rows = _group_size(out.assign(_contrast_bundle_key=bundle_key), fallback_group + ["_contrast_bundle_key"])
    detailed_signature_rows = _group_size(
        out.assign(_contrast_signature_key=service_signature_key),
        detailed_group + ["_contrast_signature_key"],
    )
    fallback_signature_rows = _group_size(
        out.assign(_contrast_signature_key=service_signature_key),
        fallback_group + ["_contrast_signature_key"],
    )

    cohort_rows = pd.Series(
        np.where(use_detailed, detailed_rows, fallback_rows),
        index=out.index,
        dtype="float64",
    )
    support_median = pd.Series(
        np.where(use_detailed, detailed_support_median, fallback_support_median),
        index=out.index,
        dtype="float64",
    )
    entertainment_median = pd.Series(
        np.where(use_detailed, detailed_entertainment_median, fallback_entertainment_median),
        index=out.index,
        dtype="float64",
    )
    paid_services_median = pd.Series(
        np.where(use_detailed, detailed_paid_services_median, fallback_paid_services_median),
        index=out.index,
        dtype="float64",
    )
    support_share_mean = pd.Series(
        np.where(use_detailed, detailed_support_share_mean, fallback_support_share_mean),
        index=out.index,
        dtype="float64",
    )
    service_mix_mean = pd.Series(
        np.where(use_detailed, detailed_service_mix_mean, fallback_service_mix_mean),
        index=out.index,
        dtype="float64",
    )
    support_rank = pd.Series(
        np.where(use_detailed, detailed_support_rank, fallback_support_rank),
        index=out.index,
        dtype="float64",
    )
    entertainment_rank = pd.Series(
        np.where(use_detailed, detailed_entertainment_rank, fallback_entertainment_rank),
        index=out.index,
        dtype="float64",
    )
    service_mix_rank = pd.Series(
        np.where(use_detailed, detailed_service_mix_rank, fallback_service_mix_rank),
        index=out.index,
        dtype="float64",
    )
    bundle_rows = pd.Series(
        np.where(use_detailed, detailed_bundle_rows, fallback_bundle_rows),
        index=out.index,
        dtype="float64",
    )
    signature_rows = pd.Series(
        np.where(use_detailed, detailed_signature_rows, fallback_signature_rows),
        index=out.index,
        dtype="float64",
    )

    out["contrast_cohort_rows"] = cohort_rows.astype("int32")
    out["contrast_used_fallback_cohort"] = (~use_detailed).astype("int8")
    out["support_count_minus_contrast_cohort"] = (support_series - support_median).astype("float64")
    out["entertainment_count_minus_contrast_cohort"] = (
        entertainment_series - entertainment_median
    ).astype("float64")
    out["paid_services_minus_contrast_cohort"] = (paid_services_series - paid_services_median).astype("float64")
    out["support_share_minus_contrast_cohort"] = (support_share_series - support_share_mean).astype("float64")
    out["service_mix_minus_contrast_cohort"] = (service_mix_series - service_mix_mean).astype("float64")
    out["support_count_rank_in_contrast_cohort"] = support_rank.astype("float64")
    out["entertainment_count_rank_in_contrast_cohort"] = entertainment_rank.astype("float64")
    out["service_mix_rank_in_contrast_cohort"] = service_mix_rank.astype("float64")
    out["bundle_archetype_rows_in_contrast_cohort"] = bundle_rows.astype("int32")
    out["bundle_archetype_share_in_contrast_cohort"] = (
        bundle_rows / cohort_rows.clip(lower=1.0)
    ).astype("float64")
    out["service_signature_rows_in_contrast_cohort"] = signature_rows.astype("int32")
    out["service_signature_share_in_contrast_cohort"] = (
        signature_rows / cohort_rows.clip(lower=1.0)
    ).astype("float64")
    out["is_rare_bundle_archetype_in_contrast_cohort"] = (
        out["bundle_archetype_share_in_contrast_cohort"] <= 0.15
    ).astype("int8")
    out["is_rare_service_signature_in_contrast_cohort"] = (
        out["service_signature_share_in_contrast_cohort"] <= 0.10
    ).astype("int8")

    return out


def _apply_block_r(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("tenure", "Contract"), block=BLOCK_R)
    out = frame.copy()

    tenure_values = pd.to_numeric(out["tenure"], errors="coerce").fillna(0.0)
    contract_text = out["Contract"].astype(str)
    contract_months = contract_text.map(
        {
            "Month-to-month": 1,
            "One year": 12,
            "Two year": 24,
        }
    ).fillna(1).astype("int16")

    out["contract_months"] = contract_months
    out["has_contract_cycle"] = (contract_months > 1).astype("int8")

    out["tenure_mod_12"] = np.mod(tenure_values, 12).astype("int16")
    out["tenure_mod_24"] = np.mod(tenure_values, 24).astype("int16")
    out["months_to_12_boundary"] = np.mod(12 - out["tenure_mod_12"], 12).astype("int16")
    out["months_to_24_boundary"] = np.mod(24 - out["tenure_mod_24"], 24).astype("int16")
    out["annual_cycle_progress"] = (out["tenure_mod_12"] / 12.0).astype("float64")
    out["two_year_cycle_progress"] = (out["tenure_mod_24"] / 24.0).astype("float64")

    contract_months_safe = contract_months.clip(lower=1)
    contract_cycle_position = np.mod(tenure_values, contract_months_safe).astype("int16")
    months_to_contract_boundary = np.mod(
        contract_months_safe - contract_cycle_position,
        contract_months_safe,
    ).astype("int16")
    months_to_contract_boundary = np.where(
        contract_months_safe > 1,
        months_to_contract_boundary,
        0,
    ).astype("int16")
    contract_cycle_progress = np.where(
        contract_months_safe > 1,
        contract_cycle_position / contract_months_safe,
        0.0,
    )

    out["contract_cycle_position"] = contract_cycle_position
    out["months_to_contract_boundary"] = months_to_contract_boundary
    out["contract_cycle_progress"] = pd.Series(contract_cycle_progress, index=out.index).astype("float64")
    out["near_contract_boundary_1"] = (
        (out["has_contract_cycle"] == 1) & (out["months_to_contract_boundary"] <= 1)
    ).astype("int8")
    out["near_contract_boundary_2"] = (
        (out["has_contract_cycle"] == 1) & (out["months_to_contract_boundary"] <= 2)
    ).astype("int8")
    out["near_contract_boundary_3"] = (
        (out["has_contract_cycle"] == 1) & (out["months_to_contract_boundary"] <= 3)
    ).astype("int8")

    out["annual_boundary_bin"] = pd.cut(
        out["months_to_12_boundary"],
        bins=[-1, 0, 2, 5, 11],
        labels=["now", "near_2", "near_5", "far"],
    ).astype(str)
    out["contract_boundary_bin"] = np.where(
        out["has_contract_cycle"] == 1,
        pd.cut(
            out["months_to_contract_boundary"],
            bins=[-1, 0, 2, 5, 24],
            labels=["now", "near_2", "near_5", "far"],
        ).astype(str),
        "no_cycle",
    )
    out["contract_x_boundary_bin"] = (
        contract_text + "__" + out["contract_boundary_bin"].astype(str)
    )
    return out


def _apply_block_s(frame: pd.DataFrame) -> pd.DataFrame:
    service_columns = (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    required = (
        "InternetService",
        "Contract",
        "PaymentMethod",
        "PhoneService",
        "MultipleLines",
    )
    _require_columns(frame, required + service_columns, block=BLOCK_S)
    out = frame.copy()

    has_internet = out["InternetService"].astype(str).str.lower().ne("no").astype("int8")
    has_phone = _yes_no_flag(out["PhoneService"])
    has_multiple_lines = _yes_no_flag(out["MultipleLines"])
    support_columns = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport")
    entertainment_columns = ("StreamingTV", "StreamingMovies")

    support_count = np.zeros(len(out), dtype=np.int16)
    entertainment_count = np.zeros(len(out), dtype=np.int16)
    for column in support_columns:
        support_count += _yes_no_flag(out[column]).to_numpy(dtype=np.int16)
    for column in entertainment_columns:
        entertainment_count += _yes_no_flag(out[column]).to_numpy(dtype=np.int16)

    paid_services_count = (
        has_internet.to_numpy(dtype=np.int16)
        + has_phone.to_numpy(dtype=np.int16)
        + has_multiple_lines.to_numpy(dtype=np.int16)
        + support_count
        + entertainment_count
    ).astype("int16")
    manual_payment = out["PaymentMethod"].astype(str).str.contains("automatic", case=False, regex=False).map(
        {True: 0, False: 1}
    ).astype("int8")

    support_signature = (
        out["OnlineSecurity"].astype(str).str.slice(0, 1).str.upper()
        + out["OnlineBackup"].astype(str).str.slice(0, 1).str.upper()
        + out["DeviceProtection"].astype(str).str.slice(0, 1).str.upper()
        + out["TechSupport"].astype(str).str.slice(0, 1).str.upper()
    )
    streaming_signature = (
        out["StreamingTV"].astype(str).str.slice(0, 1).str.upper()
        + out["StreamingMovies"].astype(str).str.slice(0, 1).str.upper()
    )

    support_count_bin = pd.cut(
        support_count,
        bins=[-1, 0, 1, 2, 4],
        labels=["0", "1", "2", "3p"],
    ).astype(str)
    entertainment_count_bin = pd.cut(
        entertainment_count,
        bins=[-1, 0, 1, 2],
        labels=["0", "1", "2"],
    ).astype(str)

    bundle_archetype = np.select(
        condlist=[
            has_internet.to_numpy(dtype=np.int8) == 0,
            (support_count == 0) & (entertainment_count == 0),
            (support_count == 0) & (entertainment_count >= 1),
            (support_count >= 1) & (entertainment_count == 0),
            support_count >= (entertainment_count + 2),
            entertainment_count >= (support_count + 1),
        ],
        choicelist=[
            "no_internet",
            "connectivity_only",
            "streaming_only",
            "support_only",
            "support_heavy",
            "entertainment_heavy",
        ],
        default="balanced",
    )

    support_total = support_count.astype("float64") + entertainment_count.astype("float64")
    support_share = np.divide(
        support_count.astype("float64"),
        np.clip(support_total, 1.0, None),
    )

    out["support_minus_entertainment"] = (support_count - entertainment_count).astype("int16")
    out["support_share_of_addons"] = pd.Series(support_share, index=out.index).astype("float64")
    out["is_support_gap_high"] = (
        (has_internet == 1) & (support_count <= 1)
    ).astype("int8")
    out["is_securityless_fiber"] = (
        out["InternetService"].astype(str).eq("Fiber optic") & (support_count == 0)
    ).astype("int8")
    out["is_manual_basic_service"] = (
        (manual_payment == 1) & (paid_services_count <= 1)
    ).astype("int8")
    out["is_manual_streaming_light_support"] = (
        (manual_payment == 1)
        & (entertainment_count >= 1)
        & (support_count <= 1)
        & (has_internet == 1)
    ).astype("int8")
    out["bundle_archetype"] = pd.Series(bundle_archetype, index=out.index).astype(str)
    out["support_signature"] = support_signature.astype(str)
    out["streaming_signature"] = streaming_signature.astype(str)
    out["service_topology"] = (
        out["InternetService"].astype(str)
        + "__"
        + out["bundle_archetype"]
        + "__"
        + out["Contract"].astype(str)
        + "__"
        + out["MultipleLines"].astype(str)
    )
    out["service_signature"] = (
        out["InternetService"].astype(str)
        + "__"
        + support_count_bin
        + "__"
        + entertainment_count_bin
        + "__"
        + support_signature
        + "__"
        + streaming_signature
    )

    return out


def _apply_block_v(frame: pd.DataFrame) -> pd.DataFrame:
    service_columns = (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    required = (
        "PaymentMethod",
        "Contract",
        "InternetService",
        "PaperlessBilling",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "MonthlyCharges",
        "TotalCharges",
    )
    _require_columns(frame, required + service_columns, block=BLOCK_V)
    out = frame.copy()

    tenure_values = pd.to_numeric(out["tenure"], errors="coerce").fillna(0.0)
    tenure_safe = tenure_values.clip(lower=1.0)
    monthly_charges = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0.0)
    total_charges = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    if "tenure_bin" not in out.columns:
        out["tenure_bin"] = _build_tenure_bin(tenure_values)

    has_internet = out["InternetService"].astype(str).str.lower().ne("no").astype("int8")
    has_phone = _yes_no_flag(out["PhoneService"])
    has_multiple_lines = _yes_no_flag(out["MultipleLines"])
    support_services_count = np.zeros(len(out), dtype=np.int16)
    entertainment_services_count = np.zeros(len(out), dtype=np.int16)
    for column in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"):
        support_services_count += _yes_no_flag(out[column]).to_numpy(dtype=np.int16)
    for column in ("StreamingTV", "StreamingMovies"):
        entertainment_services_count += _yes_no_flag(out[column]).to_numpy(dtype=np.int16)

    paid_services_count = (
        has_internet.to_numpy(dtype=np.int16)
        + has_phone.to_numpy(dtype=np.int16)
        + has_multiple_lines.to_numpy(dtype=np.int16)
        + support_services_count
        + entertainment_services_count
    ).astype("int16")
    paid_services_safe = np.clip(paid_services_count, 1, None)

    payment_friction_index = out["PaymentMethod"].astype(str).map(
        {
            "Credit card (automatic)": 0,
            "Bank transfer (automatic)": 0,
            "Electronic check": 2,
            "Mailed check": 3,
        }
    ).fillna(1).astype("int8")

    out["support_services_count"] = support_services_count.astype("int16")
    out["entertainment_services_count"] = entertainment_services_count.astype("int16")
    out["paid_services_count"] = paid_services_count.astype("int16")
    out["support_deficit_count"] = np.where(
        has_internet.to_numpy(dtype=np.int8) == 1,
        4 - support_services_count,
        0,
    ).astype("int16")
    out["monthly_per_active_service"] = (monthly_charges / paid_services_safe).astype("float64")
    out["effective_monthly_charge"] = (total_charges / tenure_safe).astype("float64")
    out["current_vs_effective_monthly_delta"] = (
        monthly_charges - out["effective_monthly_charge"]
    ).astype("float64")
    out["current_vs_effective_monthly_ratio"] = (
        monthly_charges / out["effective_monthly_charge"].clip(lower=1.0)
    ).astype("float64")
    out["payment_friction_index"] = payment_friction_index
    out["is_support_light"] = (
        (has_internet == 1) & (out["support_services_count"] <= 1)
    ).astype("int8")
    out["is_streaming_heavy_support_light"] = (
        (has_internet == 1)
        & (out["entertainment_services_count"] >= 1)
        & (out["support_services_count"] == 0)
    ).astype("int8")

    detailed_group = ["PaymentMethod", "Contract", "InternetService", "PaperlessBilling", "tenure_bin"]
    fallback_group = ["Contract", "InternetService", "tenure_bin"]
    detailed_rows = _group_size(out, detailed_group)

    detailed_monthly_median = out.groupby(detailed_group, dropna=False)["MonthlyCharges"].transform("median")
    fallback_monthly_median = out.groupby(fallback_group, dropna=False)["MonthlyCharges"].transform("median")
    detailed_mpas_median = out.groupby(detailed_group, dropna=False)["monthly_per_active_service"].transform("median")
    fallback_mpas_median = out.groupby(fallback_group, dropna=False)["monthly_per_active_service"].transform("median")
    detailed_service_median = out.groupby(detailed_group, dropna=False)["paid_services_count"].transform("median")
    fallback_service_median = out.groupby(fallback_group, dropna=False)["paid_services_count"].transform("median")
    detailed_monthly_rank = out.groupby(detailed_group, dropna=False)["MonthlyCharges"].rank(method="average", pct=True)
    fallback_monthly_rank = out.groupby(fallback_group, dropna=False)["MonthlyCharges"].rank(method="average", pct=True)

    use_detailed = detailed_rows >= 75
    cohort_monthly_median = pd.Series(
        np.where(use_detailed, detailed_monthly_median, fallback_monthly_median),
        index=out.index,
        dtype="float64",
    )
    cohort_mpas_median = pd.Series(
        np.where(use_detailed, detailed_mpas_median, fallback_mpas_median),
        index=out.index,
        dtype="float64",
    )
    cohort_service_median = pd.Series(
        np.where(use_detailed, detailed_service_median, fallback_service_median),
        index=out.index,
        dtype="float64",
    )
    cohort_monthly_rank = pd.Series(
        np.where(use_detailed, detailed_monthly_rank, fallback_monthly_rank),
        index=out.index,
        dtype="float64",
    )

    out["value_cohort_rows"] = detailed_rows
    out["value_low_support_cohort"] = (~use_detailed).astype("int8")
    out["monthly_minus_value_cohort_median"] = (monthly_charges - cohort_monthly_median).astype("float64")
    out["monthly_over_value_cohort_median"] = (
        monthly_charges / cohort_monthly_median.clip(lower=1.0)
    ).astype("float64")
    out["monthly_rank_in_value_cohort"] = cohort_monthly_rank.astype("float64")
    out["monthly_per_active_service_minus_cohort"] = (
        out["monthly_per_active_service"] - cohort_mpas_median
    ).astype("float64")
    out["paid_services_minus_cohort"] = (
        out["paid_services_count"].astype("float64") - cohort_service_median
    ).astype("float64")

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
