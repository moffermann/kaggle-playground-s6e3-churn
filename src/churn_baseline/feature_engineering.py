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
BLOCK_F: Final[str] = "F"
BLOCK_G: Final[str] = "G"
BLOCK_T: Final[str] = "T"
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
    BLOCK_F,
    BLOCK_G,
    BLOCK_T,
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
    "SURFACE": BLOCK_F,
    "FOCUS": BLOCK_F,
    "EC_SURFACE": BLOCK_F,
    "SURFACE_FIT": BLOCK_T,
    "TRAIN_SURFACE": BLOCK_T,
    "EC_SURFACE_FIT": BLOCK_T,
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
STATEFUL_BLOCKS: Final[tuple[str, ...]] = (BLOCK_G, BLOCK_T)


@dataclass(frozen=True)
class CoverageBackoffState:
    """Train-fitted support maps for family coverage/backoff features."""

    min_segment5_support: int
    segment3_counts: dict[str, int]
    segment5_counts: dict[str, int]


@dataclass(frozen=True)
class SurfaceProfile:
    """Train-fitted summary stats for one EC surface cohort."""

    rows: int
    medians: dict[str, float]
    quantile_edges: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class ECSurfaceFitState:
    """Train-only surface profiles for EC/MTM/Fiber cohorts."""

    min_detailed_support: int
    min_coarse_support: int
    detailed_profiles: dict[str, SurfaceProfile]
    coarse_profiles: dict[str, SurfaceProfile]
    paperless_profiles: dict[str, SurfaceProfile]


EC_SURFACE_FEATURE_SPECS: Final[tuple[tuple[str, str], ...]] = (
    ("monthly_charges", "monthly"),
    ("total_charges", "total"),
    ("monthly_per_active_service", "mpas"),
    ("support_services_count", "support"),
    ("paid_services_count", "paid"),
    ("current_vs_effective_monthly_delta", "delta"),
)
EC_SURFACE_QUANTILES: Final[tuple[float, ...]] = tuple(round(step / 10.0, 1) for step in range(1, 10))


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
        if block == BLOCK_F:
            out = _apply_block_f(out)
            continue
        if block == BLOCK_G:
            raise ValueError(
                "Feature block G requires train-fitted coverage state; "
                "use pipeline helpers that fit on the training split first."
            )
        if block == BLOCK_T:
            raise ValueError(
                "Feature block T requires train-fitted EC surface state; "
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


def _build_surface_tenure_bin(tenure_values: pd.Series) -> pd.Series:
    return pd.cut(
        tenure_values,
        bins=[-np.inf, 2, 6, 12, 18, 24, 36, 48, 60, np.inf],
        labels=["0_2", "3_6", "7_12", "13_18", "19_24", "25_36", "37_48", "49_60", "61_plus"],
    ).astype(str)


def _build_ec_surface_macro_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["PaymentMethod"].astype(str).eq("Electronic check")
        & frame["Contract"].astype(str).eq("Month-to-month")
        & frame["InternetService"].astype(str).eq("Fiber optic")
    )


def _ensure_tenure_bin(frame: pd.DataFrame) -> pd.Series:
    if "tenure_bin" in frame.columns:
        return frame["tenure_bin"].astype(str)
    _require_columns(frame, ("tenure",), block=BLOCK_G)
    tenure_values = pd.to_numeric(frame["tenure"], errors="coerce")
    return _build_tenure_bin(tenure_values)


def _yes_no_flag(series: pd.Series, positive_value: str = "yes") -> pd.Series:
    return series.astype(str).str.lower().eq(positive_value).astype("int8")


def ensure_monotonic_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a minimal set of numeric features for monotonic-constraint experiments."""
    _require_columns(frame, ("tenure", "Contract", "PaymentMethod"), block="MONO")
    out = frame.copy()

    if "contract_commitment_ordinal" not in out.columns:
        out["contract_commitment_ordinal"] = out["Contract"].astype(str).map(
            {
                "Month-to-month": 1,
                "One year": 12,
                "Two year": 24,
            }
        ).fillna(1).astype("int16")

    payment_method = out["PaymentMethod"].astype(str)
    payment_method_norm = payment_method.str.lower()
    if "is_manual_payment" not in out.columns:
        out["is_manual_payment"] = (~payment_method_norm.str.contains("automatic", regex=False)).astype("int8")

    if "payment_friction_index" not in out.columns:
        out["payment_friction_index"] = payment_method.map(
            {
                "Credit card (automatic)": 0,
                "Bank transfer (automatic)": 0,
                "Electronic check": 2,
                "Mailed check": 3,
            }
        ).fillna(1).astype("int8")

    return out


def _group_size(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame.groupby(list(columns), dropna=False)[columns[0]].transform("size").astype("int32")


def _compute_value_primitives(frame: pd.DataFrame, *, block: str) -> dict[str, pd.Series]:
    service_columns = (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    required = ("tenure", "InternetService", "PhoneService", "MultipleLines", "PaymentMethod", "MonthlyCharges", "TotalCharges")
    _require_columns(frame, required + service_columns, block=block)

    tenure_values = pd.to_numeric(frame["tenure"], errors="coerce").fillna(0.0)
    tenure_safe = tenure_values.clip(lower=1.0)
    monthly_charges = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").fillna(0.0)
    total_charges = pd.to_numeric(frame["TotalCharges"], errors="coerce").fillna(0.0)

    has_internet = frame["InternetService"].astype(str).str.lower().ne("no").astype("int8")
    has_phone = _yes_no_flag(frame["PhoneService"])
    has_multiple_lines = _yes_no_flag(frame["MultipleLines"])

    support_services_count = np.zeros(len(frame), dtype=np.int16)
    entertainment_services_count = np.zeros(len(frame), dtype=np.int16)
    for column in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"):
        support_services_count += _yes_no_flag(frame[column]).to_numpy(dtype=np.int16)
    for column in ("StreamingTV", "StreamingMovies"):
        entertainment_services_count += _yes_no_flag(frame[column]).to_numpy(dtype=np.int16)

    paid_services_count = (
        has_internet.to_numpy(dtype=np.int16)
        + has_phone.to_numpy(dtype=np.int16)
        + has_multiple_lines.to_numpy(dtype=np.int16)
        + support_services_count
        + entertainment_services_count
    ).astype("int16")
    paid_services_safe = np.clip(paid_services_count, 1, None)

    effective_monthly_charge = (total_charges / tenure_safe).astype("float64")
    monthly_per_active_service = (monthly_charges / paid_services_safe).astype("float64")
    payment_friction_index = frame["PaymentMethod"].astype(str).map(
        {
            "Credit card (automatic)": 0,
            "Bank transfer (automatic)": 0,
            "Electronic check": 2,
            "Mailed check": 3,
        }
    ).fillna(1).astype("int8")

    return {
        "tenure_values": tenure_values.astype("float64"),
        "tenure_safe": tenure_safe.astype("float64"),
        "monthly_charges": monthly_charges.astype("float64"),
        "total_charges": total_charges.astype("float64"),
        "has_internet": has_internet.astype("int8"),
        "support_services_count": pd.Series(support_services_count, index=frame.index, dtype="int16"),
        "entertainment_services_count": pd.Series(entertainment_services_count, index=frame.index, dtype="int16"),
        "paid_services_count": pd.Series(paid_services_count, index=frame.index, dtype="int16"),
        "support_deficit_count": pd.Series(
            np.where(has_internet.to_numpy(dtype=np.int8) == 1, 4 - support_services_count, 0),
            index=frame.index,
            dtype="int16",
        ),
        "monthly_per_active_service": pd.Series(monthly_per_active_service, index=frame.index, dtype="float64"),
        "effective_monthly_charge": pd.Series(effective_monthly_charge, index=frame.index, dtype="float64"),
        "current_vs_effective_monthly_delta": pd.Series(
            monthly_charges - effective_monthly_charge,
            index=frame.index,
            dtype="float64",
        ),
        "current_vs_effective_monthly_ratio": pd.Series(
            monthly_charges / np.clip(effective_monthly_charge, 1.0, None),
            index=frame.index,
            dtype="float64",
        ),
        "payment_friction_index": payment_friction_index.astype("int8"),
    }


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


def _compute_quantile_edges(
    values: pd.Series,
    *,
    quantiles: Sequence[float] = EC_SURFACE_QUANTILES,
) -> tuple[float, ...]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return ()
    edges = numeric.quantile(list(quantiles)).astype("float64").to_numpy()
    if edges.size == 0:
        return ()
    unique_edges = np.unique(edges[np.isfinite(edges)])
    return tuple(float(edge) for edge in unique_edges.tolist())


def _fit_surface_profiles(
    keys: pd.Series,
    values: pd.DataFrame,
) -> dict[str, SurfaceProfile]:
    profiles: dict[str, SurfaceProfile] = {}
    for key, group in values.groupby(keys.astype(str), dropna=False):
        medians: dict[str, float] = {}
        quantile_edges: dict[str, tuple[float, ...]] = {}
        for source_column, _ in EC_SURFACE_FEATURE_SPECS:
            series = pd.to_numeric(group[source_column], errors="coerce").fillna(0.0).astype("float64")
            medians[source_column] = float(series.median())
            quantile_edges[source_column] = _compute_quantile_edges(series)
        profiles[str(key)] = SurfaceProfile(
            rows=int(len(group)),
            medians=medians,
            quantile_edges=quantile_edges,
        )
    return profiles


def _approx_rank_from_edges(
    values: pd.Series,
    edge_series: pd.Series,
) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0).astype("float64").to_numpy()
    ranks = np.full(len(values), 0.5, dtype="float64")
    for idx, (value, edges) in enumerate(zip(numeric_values, edge_series.tolist())):
        if not isinstance(edges, (tuple, list, np.ndarray)) or len(edges) == 0:
            continue
        edges_arr = np.asarray(edges, dtype="float64")
        if edges_arr.size == 0:
            continue
        ranks[idx] = (np.searchsorted(edges_arr, value, side="right") + 0.5) / float(edges_arr.size + 1)
    return pd.Series(ranks, index=values.index, dtype="float64")


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


def fit_ec_surface_state(
    frame: pd.DataFrame,
    *,
    min_detailed_support: int = 75,
    min_coarse_support: int = 75,
) -> ECSurfaceFitState:
    """Fit train-only surface profiles for EC/MTM/Fiber cohorts."""
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
    _require_columns(frame, required + service_columns, block=BLOCK_T)

    primitives = _compute_value_primitives(frame, block=BLOCK_T)
    tenure_values = primitives["tenure_values"]
    tenure_bin = _ensure_tenure_bin(frame)
    macro_mask = _build_ec_surface_macro_mask(frame)
    if not bool(macro_mask.any()):
        return ECSurfaceFitState(
            min_detailed_support=int(min_detailed_support),
            min_coarse_support=int(min_coarse_support),
            detailed_profiles={},
            coarse_profiles={},
            paperless_profiles={},
        )

    fine_tenure_bin = _build_surface_tenure_bin(tenure_values)
    macro_df = pd.DataFrame(
        {
            "_paperless": frame.loc[macro_mask, "PaperlessBilling"].astype(str),
            "_fine_tenure": fine_tenure_bin.loc[macro_mask].astype(str),
            "_coarse_tenure": tenure_bin.loc[macro_mask].astype(str),
            "monthly_charges": primitives["monthly_charges"].loc[macro_mask].astype("float64"),
            "total_charges": primitives["total_charges"].loc[macro_mask].astype("float64"),
            "monthly_per_active_service": primitives["monthly_per_active_service"].loc[macro_mask].astype("float64"),
            "support_services_count": primitives["support_services_count"].loc[macro_mask].astype("float64"),
            "paid_services_count": primitives["paid_services_count"].loc[macro_mask].astype("float64"),
            "current_vs_effective_monthly_delta": primitives["current_vs_effective_monthly_delta"].loc[macro_mask].astype("float64"),
        },
        index=frame.index[macro_mask],
    )

    detailed_key = macro_df["_paperless"] + "__" + macro_df["_fine_tenure"]
    coarse_key = macro_df["_paperless"] + "__" + macro_df["_coarse_tenure"]
    paperless_key = macro_df["_paperless"]

    return ECSurfaceFitState(
        min_detailed_support=int(min_detailed_support),
        min_coarse_support=int(min_coarse_support),
        detailed_profiles=_fit_surface_profiles(detailed_key, macro_df),
        coarse_profiles=_fit_surface_profiles(coarse_key, macro_df),
        paperless_profiles=_fit_surface_profiles(paperless_key, macro_df),
    )


def apply_ec_surface_fit_features(
    frame: pd.DataFrame,
    state: ECSurfaceFitState,
) -> pd.DataFrame:
    """Append train-fitted EC surface features to any frame."""
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
    _require_columns(frame, required + service_columns, block=BLOCK_T)

    out = frame.copy()
    primitives = _compute_value_primitives(out, block=BLOCK_T)
    tenure_values = primitives["tenure_values"]
    if "tenure_bin" not in out.columns:
        out["tenure_bin"] = _build_tenure_bin(tenure_values)

    macro_mask = _build_ec_surface_macro_mask(out)
    fine_tenure_bin = _build_surface_tenure_bin(tenure_values)

    out["ec_surface_fit_is_family"] = macro_mask.astype("int8")
    out["ec_surface_fit_is_paperless_family"] = (
        macro_mask & out["PaperlessBilling"].astype(str).eq("Yes")
    ).astype("int8")
    out["ec_surface_fit_level"] = "other"
    out["ec_surface_fit_segment"] = "other"
    out["ec_surface_fit_rows"] = 0
    out["ec_surface_fit_support_bucket"] = "zero"
    out["ec_surface_fit_used_coarse"] = 0
    out["ec_surface_fit_used_paperless"] = 0
    out["ec_surface_fit_monthly_minus_train_median"] = 0.0
    out["ec_surface_fit_monthly_over_train_median"] = 1.0
    out["ec_surface_fit_monthly_rank_fit"] = 0.5
    out["ec_surface_fit_total_minus_train_median"] = 0.0
    out["ec_surface_fit_total_rank_fit"] = 0.5
    out["ec_surface_fit_mpas_minus_train_median"] = 0.0
    out["ec_surface_fit_mpas_rank_fit"] = 0.5
    out["ec_surface_fit_paid_minus_train_median"] = 0.0
    out["ec_surface_fit_support_minus_train_median"] = 0.0
    out["ec_surface_fit_delta_minus_train_median"] = 0.0
    out["ec_surface_fit_delta_rank_fit"] = 0.5
    out["ec_surface_fit_pressure_x_support_deficit"] = 0.0

    if not bool(macro_mask.any()) or not state.paperless_profiles:
        return out

    macro_df = pd.DataFrame(
        {
            "_paperless": out.loc[macro_mask, "PaperlessBilling"].astype(str),
            "_fine_tenure": fine_tenure_bin.loc[macro_mask].astype(str),
            "_coarse_tenure": out.loc[macro_mask, "tenure_bin"].astype(str),
            "monthly_charges": primitives["monthly_charges"].loc[macro_mask].astype("float64"),
            "total_charges": primitives["total_charges"].loc[macro_mask].astype("float64"),
            "monthly_per_active_service": primitives["monthly_per_active_service"].loc[macro_mask].astype("float64"),
            "support_services_count": primitives["support_services_count"].loc[macro_mask].astype("float64"),
            "paid_services_count": primitives["paid_services_count"].loc[macro_mask].astype("float64"),
            "current_vs_effective_monthly_delta": primitives["current_vs_effective_monthly_delta"].loc[macro_mask].astype("float64"),
            "support_deficit_count": primitives["support_deficit_count"].loc[macro_mask].astype("float64"),
        },
        index=out.index[macro_mask],
    )

    detailed_key = macro_df["_paperless"] + "__" + macro_df["_fine_tenure"]
    coarse_key = macro_df["_paperless"] + "__" + macro_df["_coarse_tenure"]
    paperless_key = macro_df["_paperless"]

    detailed_rows_map = {key: profile.rows for key, profile in state.detailed_profiles.items()}
    coarse_rows_map = {key: profile.rows for key, profile in state.coarse_profiles.items()}
    paperless_rows_map = {key: profile.rows for key, profile in state.paperless_profiles.items()}

    detailed_rows = detailed_key.map(detailed_rows_map).fillna(0).astype("int32")
    coarse_rows = coarse_key.map(coarse_rows_map).fillna(0).astype("int32")
    paperless_rows = paperless_key.map(paperless_rows_map).fillna(0).astype("int32")

    use_detailed = detailed_rows.ge(int(state.min_detailed_support))
    use_coarse = (~use_detailed) & coarse_rows.ge(int(state.min_coarse_support))
    selected_level = pd.Series(
        np.where(use_detailed, "detail", np.where(use_coarse, "coarse", "paperless")),
        index=macro_df.index,
        dtype="object",
    )
    selected_segment = pd.Series(
        np.where(
            use_detailed,
            "detail__" + detailed_key.astype(str),
            np.where(
                use_coarse,
                "coarse__" + coarse_key.astype(str),
                "paperless__" + paperless_key.astype(str),
            ),
        ),
        index=macro_df.index,
        dtype="object",
    )
    selected_rows = pd.Series(
        np.where(use_detailed, detailed_rows, np.where(use_coarse, coarse_rows, paperless_rows)),
        index=macro_df.index,
        dtype="int32",
    )

    out.loc[macro_df.index, "ec_surface_fit_level"] = selected_level.astype(str)
    out.loc[macro_df.index, "ec_surface_fit_segment"] = selected_segment.astype(str)
    out.loc[macro_df.index, "ec_surface_fit_rows"] = selected_rows.astype("int32")
    out.loc[macro_df.index, "ec_surface_fit_support_bucket"] = _support_bucket(selected_rows)
    out.loc[macro_df.index, "ec_surface_fit_used_coarse"] = use_coarse.astype("int8")
    out.loc[macro_df.index, "ec_surface_fit_used_paperless"] = (~use_detailed & ~use_coarse).astype("int8")

    combined_profiles: dict[str, SurfaceProfile] = {}
    combined_profiles.update({f"detail__{key}": profile for key, profile in state.detailed_profiles.items()})
    combined_profiles.update({f"coarse__{key}": profile for key, profile in state.coarse_profiles.items()})
    combined_profiles.update({f"paperless__{key}": profile for key, profile in state.paperless_profiles.items()})

    for source_column, prefix in EC_SURFACE_FEATURE_SPECS:
        median_map = {
            key: profile.medians[source_column]
            for key, profile in combined_profiles.items()
        }
        edge_map = {
            key: profile.quantile_edges[source_column]
            for key, profile in combined_profiles.items()
        }
        selected_median = selected_segment.map(median_map)
        selected_edges = selected_segment.map(edge_map)

        selected_median = selected_median.fillna(macro_df[source_column]).astype("float64")
        selected_rank = _approx_rank_from_edges(macro_df[source_column], selected_edges)
        minus_series = (macro_df[source_column] - selected_median).astype("float64")

        if prefix == "monthly":
            out.loc[macro_df.index, "ec_surface_fit_monthly_minus_train_median"] = minus_series
            out.loc[macro_df.index, "ec_surface_fit_monthly_over_train_median"] = (
                macro_df[source_column] / selected_median.clip(lower=1.0)
            ).astype("float64")
            out.loc[macro_df.index, "ec_surface_fit_monthly_rank_fit"] = selected_rank.astype("float64")
        elif prefix == "total":
            out.loc[macro_df.index, "ec_surface_fit_total_minus_train_median"] = minus_series
            out.loc[macro_df.index, "ec_surface_fit_total_rank_fit"] = selected_rank.astype("float64")
        elif prefix == "mpas":
            out.loc[macro_df.index, "ec_surface_fit_mpas_minus_train_median"] = minus_series
            out.loc[macro_df.index, "ec_surface_fit_mpas_rank_fit"] = selected_rank.astype("float64")
        elif prefix == "paid":
            out.loc[macro_df.index, "ec_surface_fit_paid_minus_train_median"] = minus_series
        elif prefix == "support":
            out.loc[macro_df.index, "ec_surface_fit_support_minus_train_median"] = minus_series
        elif prefix == "delta":
            out.loc[macro_df.index, "ec_surface_fit_delta_minus_train_median"] = minus_series
            out.loc[macro_df.index, "ec_surface_fit_delta_rank_fit"] = selected_rank.astype("float64")

    out.loc[macro_df.index, "ec_surface_fit_pressure_x_support_deficit"] = (
        out.loc[macro_df.index, "ec_surface_fit_monthly_over_train_median"].astype("float64")
        * macro_df["support_deficit_count"].astype("float64")
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

    primitives = _compute_value_primitives(out, block=BLOCK_V)
    tenure_values = primitives["tenure_values"]
    monthly_charges = primitives["monthly_charges"]
    if "tenure_bin" not in out.columns:
        out["tenure_bin"] = _build_tenure_bin(tenure_values)

    has_internet = primitives["has_internet"]
    out["support_services_count"] = primitives["support_services_count"].astype("int16")
    out["entertainment_services_count"] = primitives["entertainment_services_count"].astype("int16")
    out["paid_services_count"] = primitives["paid_services_count"].astype("int16")
    out["support_deficit_count"] = primitives["support_deficit_count"].astype("int16")
    out["monthly_per_active_service"] = primitives["monthly_per_active_service"].astype("float64")
    out["effective_monthly_charge"] = primitives["effective_monthly_charge"].astype("float64")
    out["current_vs_effective_monthly_delta"] = primitives["current_vs_effective_monthly_delta"].astype("float64")
    out["current_vs_effective_monthly_ratio"] = primitives["current_vs_effective_monthly_ratio"].astype("float64")
    out["payment_friction_index"] = primitives["payment_friction_index"].astype("int8")
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


def _apply_block_f(frame: pd.DataFrame) -> pd.DataFrame:
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
    _require_columns(frame, required + service_columns, block=BLOCK_F)
    out = frame.copy()

    primitives = _compute_value_primitives(out, block=BLOCK_F)
    tenure_values = primitives["tenure_values"]
    monthly_charges = primitives["monthly_charges"]
    total_charges = primitives["total_charges"]
    if "tenure_bin" not in out.columns:
        out["tenure_bin"] = _build_tenure_bin(tenure_values)

    macro_mask = (
        out["PaymentMethod"].astype(str).eq("Electronic check")
        & out["Contract"].astype(str).eq("Month-to-month")
        & out["InternetService"].astype(str).eq("Fiber optic")
    )
    fine_tenure_bin = _build_surface_tenure_bin(tenure_values)
    out["ec_surface_is_family"] = macro_mask.astype("int8")
    out["ec_surface_is_paperless_family"] = (
        macro_mask & out["PaperlessBilling"].astype(str).eq("Yes")
    ).astype("int8")
    out["ec_surface_tenure_bin_fine"] = pd.Series(
        np.where(macro_mask, fine_tenure_bin, "other"),
        index=out.index,
        dtype="object",
    ).astype(str)
    out["ec_surface_segment"] = pd.Series(
        np.where(
            macro_mask,
            out["PaperlessBilling"].astype(str) + "__" + out["ec_surface_tenure_bin_fine"].astype(str),
            "other",
        ),
        index=out.index,
        dtype="object",
    ).astype(str)

    out["ec_surface_rows"] = 0
    out["ec_surface_used_fallback"] = 0
    out["ec_surface_monthly_minus_median"] = 0.0
    out["ec_surface_monthly_over_median"] = 1.0
    out["ec_surface_monthly_rank"] = 0.5
    out["ec_surface_total_minus_median"] = 0.0
    out["ec_surface_total_rank"] = 0.5
    out["ec_surface_mpas_minus_median"] = 0.0
    out["ec_surface_mpas_rank"] = 0.5
    out["ec_surface_paid_minus_median"] = 0.0
    out["ec_surface_support_minus_median"] = 0.0
    out["ec_surface_delta_minus_median"] = 0.0
    out["ec_surface_delta_rank"] = 0.5
    out["ec_surface_pressure_x_support_deficit"] = 0.0

    if not macro_mask.any():
        return out

    macro_df = pd.DataFrame(
        {
            "_paperless": out.loc[macro_mask, "PaperlessBilling"].astype(str),
            "_fine_tenure": fine_tenure_bin.loc[macro_mask].astype(str),
            "_coarse_tenure": out.loc[macro_mask, "tenure_bin"].astype(str),
            "monthly_charges": monthly_charges.loc[macro_mask].astype("float64"),
            "total_charges": total_charges.loc[macro_mask].astype("float64"),
            "monthly_per_active_service": primitives["monthly_per_active_service"].loc[macro_mask].astype("float64"),
            "support_services_count": primitives["support_services_count"].loc[macro_mask].astype("float64"),
            "paid_services_count": primitives["paid_services_count"].loc[macro_mask].astype("float64"),
            "current_vs_effective_monthly_delta": primitives["current_vs_effective_monthly_delta"].loc[macro_mask].astype("float64"),
            "support_deficit_count": primitives["support_deficit_count"].loc[macro_mask].astype("float64"),
        },
        index=out.index[macro_mask],
    )
    detailed_group = ["_paperless", "_fine_tenure"]
    fallback_group = ["_paperless", "_coarse_tenure"]
    detailed_rows = macro_df.groupby(detailed_group, dropna=False)["_paperless"].transform("size")
    fallback_rows = macro_df.groupby(fallback_group, dropna=False)["_paperless"].transform("size")
    use_detailed = detailed_rows >= 75

    macro_features: dict[str, pd.Series] = {
        "rows": pd.Series(np.where(use_detailed, detailed_rows, fallback_rows), index=macro_df.index, dtype="float64"),
        "used_fallback": pd.Series((~use_detailed).astype("int8"), index=macro_df.index, dtype="int8"),
    }
    for source_column, prefix in EC_SURFACE_FEATURE_SPECS:
        detailed_median = macro_df.groupby(detailed_group, dropna=False)[source_column].transform("median")
        fallback_median = macro_df.groupby(fallback_group, dropna=False)[source_column].transform("median")
        detailed_rank = macro_df.groupby(detailed_group, dropna=False)[source_column].rank(method="average", pct=True)
        fallback_rank = macro_df.groupby(fallback_group, dropna=False)[source_column].rank(method="average", pct=True)

        cohort_median = pd.Series(
            np.where(use_detailed, detailed_median, fallback_median),
            index=macro_df.index,
            dtype="float64",
        )
        cohort_rank = pd.Series(
            np.where(use_detailed, detailed_rank, fallback_rank),
            index=macro_df.index,
            dtype="float64",
        )
        macro_features[f"{prefix}_minus_median"] = (macro_df[source_column] - cohort_median).astype("float64")
        macro_features[f"{prefix}_rank"] = cohort_rank.astype("float64")
        if prefix == "monthly":
            macro_features["monthly_over_median"] = (
                macro_df[source_column] / cohort_median.clip(lower=1.0)
            ).astype("float64")

    out.loc[macro_df.index, "ec_surface_rows"] = macro_features["rows"].astype("int32")
    out.loc[macro_df.index, "ec_surface_used_fallback"] = macro_features["used_fallback"].astype("int8")
    out.loc[macro_df.index, "ec_surface_monthly_minus_median"] = macro_features["monthly_minus_median"].astype("float64")
    out.loc[macro_df.index, "ec_surface_monthly_over_median"] = macro_features["monthly_over_median"].astype("float64")
    out.loc[macro_df.index, "ec_surface_monthly_rank"] = macro_features["monthly_rank"].astype("float64")
    out.loc[macro_df.index, "ec_surface_total_minus_median"] = macro_features["total_minus_median"].astype("float64")
    out.loc[macro_df.index, "ec_surface_total_rank"] = macro_features["total_rank"].astype("float64")
    out.loc[macro_df.index, "ec_surface_mpas_minus_median"] = macro_features["mpas_minus_median"].astype("float64")
    out.loc[macro_df.index, "ec_surface_mpas_rank"] = macro_features["mpas_rank"].astype("float64")
    out.loc[macro_df.index, "ec_surface_paid_minus_median"] = macro_features["paid_minus_median"].astype("float64")
    out.loc[macro_df.index, "ec_surface_support_minus_median"] = macro_features["support_minus_median"].astype("float64")
    out.loc[macro_df.index, "ec_surface_delta_minus_median"] = macro_features["delta_minus_median"].astype("float64")
    out.loc[macro_df.index, "ec_surface_delta_rank"] = macro_features["delta_rank"].astype("float64")
    out.loc[macro_df.index, "ec_surface_pressure_x_support_deficit"] = (
        macro_features["monthly_over_median"] * macro_df["support_deficit_count"]
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
