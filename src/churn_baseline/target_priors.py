"""Fold-safe hierarchical target priors for churn experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PRIOR_TENURE_BIN_COLUMN = "prior_tenure_bin"
PRIOR_MONTHLY_PER_TENURE_COLUMN = "prior_monthly_per_tenure"
PRIOR_ADDONS_ACTIVE_COUNT_COLUMN = "prior_addons_active_count"
PRIOR_KEY_SEPARATOR = "__"
PRIOR_EPSILON = 1e-6
SERVICE_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)


@dataclass(frozen=True)
class PriorSpec:
    """Definition for one grouped target prior."""

    name: str
    columns: tuple[str, ...]
    min_support: int = 50
    smoothing_alpha: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "min_support": int(self.min_support),
            "smoothing_alpha": float(self.smoothing_alpha),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PriorSpec":
        return cls(
            name=str(payload["name"]),
            columns=tuple(str(v) for v in payload["columns"]),
            min_support=int(payload["min_support"]),
            smoothing_alpha=float(payload["smoothing_alpha"]),
        )


def build_default_prior_specs() -> tuple[PriorSpec, ...]:
    """Return a coarse-to-fine hierarchy of target priors."""
    return (
        PriorSpec(name="contract", columns=("Contract",), min_support=1, smoothing_alpha=300.0),
        PriorSpec(
            name="contract_internet",
            columns=("Contract", "InternetService"),
            min_support=20,
            smoothing_alpha=120.0,
        ),
        PriorSpec(
            name="segment3",
            columns=("PaymentMethod", "Contract", "InternetService"),
            min_support=30,
            smoothing_alpha=60.0,
        ),
        PriorSpec(
            name="segment4_tenure",
            columns=("PaymentMethod", "Contract", "InternetService", PRIOR_TENURE_BIN_COLUMN),
            min_support=50,
            smoothing_alpha=35.0,
        ),
        PriorSpec(
            name="segment5_paperless",
            columns=(
                "PaymentMethod",
                "Contract",
                "InternetService",
                "PaperlessBilling",
                PRIOR_TENURE_BIN_COLUMN,
            ),
            min_support=80,
            smoothing_alpha=25.0,
        ),
    )


def build_default_numeric_deviation_columns() -> tuple[str, ...]:
    """Return numeric columns used for fold-safe cohort means and deviations."""
    return (
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        PRIOR_MONTHLY_PER_TENURE_COLUMN,
        PRIOR_ADDONS_ACTIVE_COUNT_COLUMN,
    )


def prepare_prior_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic columns needed by grouped target priors."""
    out = features.copy()
    tenure_values = pd.to_numeric(out["tenure"], errors="coerce")
    tenure_safe = tenure_values.clip(lower=1).fillna(1.0)
    monthly_charges = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0.0)
    total_charges = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)

    if PRIOR_TENURE_BIN_COLUMN not in out.columns:
        out[PRIOR_TENURE_BIN_COLUMN] = pd.cut(
            tenure_values,
            bins=[-np.inf, 6, 12, 24, 48, np.inf],
            labels=["0_6", "7_12", "13_24", "25_48", "49_plus"],
        ).astype(str)
    out[PRIOR_TENURE_BIN_COLUMN] = out[PRIOR_TENURE_BIN_COLUMN].replace("nan", "__MISSING__")

    if PRIOR_MONTHLY_PER_TENURE_COLUMN not in out.columns:
        out[PRIOR_MONTHLY_PER_TENURE_COLUMN] = monthly_charges / tenure_safe

    if PRIOR_ADDONS_ACTIVE_COUNT_COLUMN not in out.columns:
        if "addons_active_count" in out.columns:
            out[PRIOR_ADDONS_ACTIVE_COUNT_COLUMN] = (
                pd.to_numeric(out["addons_active_count"], errors="coerce").fillna(0.0).astype("float64")
            )
        else:
            _require_columns(out, SERVICE_COLUMNS)
            addons_active = np.zeros(len(out), dtype=np.int16)
            for column in SERVICE_COLUMNS:
                addons_active += out[column].astype(str).str.lower().eq("yes").to_numpy(dtype=np.int16)
            out[PRIOR_ADDONS_ACTIVE_COUNT_COLUMN] = addons_active.astype("float64")
    return out


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Target prior columns not found in dataframe: {missing}")


def _build_group_keys(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    _require_columns(frame, columns)
    key = frame[columns[0]].astype(str)
    for column_name in columns[1:]:
        key = key + PRIOR_KEY_SEPARATOR + frame[column_name].astype(str)
    return key


def _safe_logit(values: pd.Series | np.ndarray | float) -> pd.Series:
    series = pd.Series(values, copy=False, dtype="float64")
    clipped = series.clip(lower=PRIOR_EPSILON, upper=1.0 - PRIOR_EPSILON)
    return np.log(clipped / (1.0 - clipped))


class HierarchicalTargetPriorEncoder:
    """Fit grouped target priors on train folds and reuse them safely."""

    def __init__(
        self,
        specs: Sequence[PriorSpec] | None = None,
        *,
        include_numeric_deviation: bool = False,
        numeric_columns: Sequence[str] | None = None,
    ) -> None:
        normalized_specs = tuple(specs or build_default_prior_specs())
        if not normalized_specs:
            raise ValueError("At least one prior spec is required.")
        self.specs = normalized_specs
        self.include_numeric_deviation = bool(include_numeric_deviation)
        if numeric_columns is None:
            numeric_columns = (
                build_default_numeric_deviation_columns() if self.include_numeric_deviation else ()
            )
        self.numeric_columns = tuple(str(value) for value in numeric_columns)
        self.global_mean_: float | None = None
        self.stats_by_spec_: dict[str, dict[str, dict[str, float | int]]] | None = None
        self.numeric_global_means_: dict[str, float] | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "HierarchicalTargetPriorEncoder":
        work = prepare_prior_frame(features)
        target_num = pd.to_numeric(target, errors="coerce").astype("float64")
        if len(work) != len(target_num):
            raise ValueError("features and target must have the same number of rows.")

        stats_by_spec: dict[str, dict[str, dict[str, float | int]]] = {}
        self.global_mean_ = float(target_num.mean())
        self.numeric_global_means_ = {}

        numeric_series_by_column: dict[str, pd.Series] = {}
        for column_name in self.numeric_columns:
            numeric_series = pd.to_numeric(work[column_name], errors="coerce").astype("float64")
            global_numeric_mean = float(numeric_series.mean())
            self.numeric_global_means_[column_name] = global_numeric_mean
            numeric_series_by_column[column_name] = numeric_series.fillna(global_numeric_mean)

        for spec in self.specs:
            grouped = work[list(spec.columns)].copy()
            grouped["__target__"] = target_num.values
            aggregations: dict[str, tuple[str, str]] = {
                "target_sum": ("__target__", "sum"),
                "count": ("__target__", "count"),
            }
            for column_name, series in numeric_series_by_column.items():
                grouped[f"__num__{column_name}"] = series.values
                aggregations[f"numeric_sum__{column_name}"] = (f"__num__{column_name}", "sum")

            summary = grouped.groupby(list(spec.columns), dropna=False).agg(**aggregations).reset_index()
            mapping: dict[str, dict[str, float | int]] = {}
            for row in summary.to_dict(orient="records"):
                key_values = [row[column_name] for column_name in spec.columns]
                key = PRIOR_KEY_SEPARATOR.join(str(value) for value in key_values)
                mapping[key] = {
                    "sum": float(row["target_sum"]),
                    "count": int(row["count"]),
                }
                if self.include_numeric_deviation and self.numeric_columns:
                    mapping[key]["numeric_sums"] = {
                        column_name: float(row[f"numeric_sum__{column_name}"])
                        for column_name in self.numeric_columns
                    }
            stats_by_spec[spec.name] = mapping

        self.stats_by_spec_ = stats_by_spec
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None or self.stats_by_spec_ is None:
            raise RuntimeError("HierarchicalTargetPriorEncoder must be fitted before transform().")
        if self.include_numeric_deviation and self.numeric_global_means_ is None:
            raise RuntimeError("Numeric deviation metadata is missing from fitted encoder.")

        work = prepare_prior_frame(features)
        out = pd.DataFrame(index=work.index)

        base_global = float(self.global_mean_)
        base_global_logit = float(_safe_logit([base_global]).iloc[0])
        out["prior_global"] = base_global
        out["prior_global_logit"] = base_global_logit

        hier_prior = pd.Series(base_global, index=work.index, dtype="float64")
        hier_support = pd.Series(0, index=work.index, dtype="int32")
        hier_source = pd.Series("global", index=work.index, dtype="object")
        hier_level = pd.Series(0, index=work.index, dtype="int8")
        numeric_mean_by_spec: dict[str, dict[str, pd.Series]] = {}

        spec_order = list(self.specs)
        for level, spec in enumerate(spec_order, start=1):
            stats = self.stats_by_spec_[spec.name]
            keys = _build_group_keys(work, spec.columns)
            sums = keys.map({key: float(value["sum"]) for key, value in stats.items()}).fillna(0.0)
            counts = (
                keys.map({key: int(value["count"]) for key, value in stats.items()}).fillna(0).astype("int32")
            )
            posterior = (sums + float(spec.smoothing_alpha) * base_global) / (
                counts.astype("float64") + float(spec.smoothing_alpha)
            )

            out[f"prior_{spec.name}"] = posterior.astype("float64")
            out[f"prior_logit_{spec.name}"] = _safe_logit(posterior).astype("float64")
            out[f"support_{spec.name}"] = counts
            out[f"is_low_support_{spec.name}"] = (counts < int(spec.min_support)).astype("int8")

            if self.include_numeric_deviation and self.numeric_columns:
                numeric_mean_by_spec[spec.name] = {}
                for column_name in self.numeric_columns:
                    numeric_sums = keys.map(
                        {
                            key: float(
                                dict(value.get("numeric_sums", {})).get(
                                    column_name,
                                    self.numeric_global_means_[column_name],
                                )
                            )
                            for key, value in stats.items()
                        }
                    ).fillna(0.0)
                    global_numeric_mean = float(self.numeric_global_means_[column_name])
                    numeric_mean = (numeric_sums + float(spec.smoothing_alpha) * global_numeric_mean) / (
                        counts.astype("float64") + float(spec.smoothing_alpha)
                    )
                    numeric_mean_by_spec[spec.name][column_name] = numeric_mean.astype("float64")

        for reverse_level, spec in enumerate(reversed(spec_order), start=1):
            prior_column = f"prior_{spec.name}"
            support_column = f"support_{spec.name}"
            valid_mask = out[support_column] >= int(spec.min_support)
            assign_mask = valid_mask & hier_source.eq("global")
            if not bool(assign_mask.any()):
                continue
            hier_prior.loc[assign_mask] = out.loc[assign_mask, prior_column]
            hier_support.loc[assign_mask] = out.loc[assign_mask, support_column]
            hier_source.loc[assign_mask] = spec.name
            hier_level.loc[assign_mask] = int(len(spec_order) - reverse_level + 1)

        out["prior_hier"] = hier_prior.astype("float64")
        out["prior_hier_logit"] = _safe_logit(hier_prior).astype("float64")
        out["prior_hier_support"] = hier_support.astype("int32")
        out["prior_hier_source"] = hier_source.astype(str)
        out["prior_hier_level"] = hier_level.astype("int8")
        out["prior_hier_is_global"] = hier_source.eq("global").astype("int8")

        if self.include_numeric_deviation and self.numeric_columns:
            hier_numeric_means: dict[str, pd.Series] = {
                column_name: pd.Series(
                    float(self.numeric_global_means_[column_name]),
                    index=work.index,
                    dtype="float64",
                )
                for column_name in self.numeric_columns
            }
            for spec in reversed(spec_order):
                support_column = f"support_{spec.name}"
                assign_mask = out["prior_hier_source"].eq(spec.name) & (
                    out[support_column] >= int(spec.min_support)
                )
                if not bool(assign_mask.any()):
                    continue
                for column_name in self.numeric_columns:
                    hier_numeric_means[column_name].loc[assign_mask] = numeric_mean_by_spec[spec.name][
                        column_name
                    ].loc[assign_mask]

            for column_name in self.numeric_columns:
                base_values = pd.to_numeric(work[column_name], errors="coerce").astype("float64")
                global_numeric_mean = float(self.numeric_global_means_[column_name])
                base_values = base_values.fillna(global_numeric_mean)
                cohort_mean = hier_numeric_means[column_name]
                delta_values = base_values - cohort_mean
                out[f"cohort_mean_{column_name}"] = cohort_mean.astype("float64")
                out[f"cohort_delta_{column_name}"] = delta_values.astype("float64")
                out[f"cohort_rel_{column_name}"] = (
                    delta_values / (cohort_mean.abs() + 1.0)
                ).astype("float64")
        return out

    def fit_transform(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        return self.fit(features, target).transform(features)

    def to_dict(self) -> dict[str, Any]:
        if self.global_mean_ is None or self.stats_by_spec_ is None:
            raise RuntimeError("HierarchicalTargetPriorEncoder is not fitted.")
        return {
            "global_mean": float(self.global_mean_),
            "specs": [spec.to_dict() for spec in self.specs],
            "stats_by_spec": self.stats_by_spec_,
            "include_numeric_deviation": self.include_numeric_deviation,
            "numeric_columns": list(self.numeric_columns),
            "numeric_global_means": dict(self.numeric_global_means_ or {}),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HierarchicalTargetPriorEncoder":
        encoder = cls(
            specs=[PriorSpec.from_dict(item) for item in payload["specs"]],
            include_numeric_deviation=bool(payload.get("include_numeric_deviation", False)),
            numeric_columns=payload.get("numeric_columns"),
        )
        encoder.global_mean_ = float(payload["global_mean"])
        encoder.stats_by_spec_ = {
            str(spec_name): {
                str(key): {
                    "sum": float(values["sum"]),
                    "count": int(values["count"]),
                    "numeric_sums": {
                        str(column_name): float(column_value)
                        for column_name, column_value in dict(values.get("numeric_sums", {})).items()
                    },
                }
                for key, values in dict(spec_stats).items()
            }
            for spec_name, spec_stats in dict(payload["stats_by_spec"]).items()
        }
        encoder.numeric_global_means_ = {
            str(column_name): float(column_value)
            for column_name, column_value in dict(payload.get("numeric_global_means", {})).items()
        }
        return encoder


def save_target_prior_encoder(
    encoder: HierarchicalTargetPriorEncoder,
    path: str | Path,
) -> Path:
    """Persist target prior encoder metadata to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(encoder.to_dict(), indent=2), encoding="utf-8")
    return out_path


def load_target_prior_encoder(path: str | Path) -> HierarchicalTargetPriorEncoder:
    """Load target prior encoder metadata from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HierarchicalTargetPriorEncoder.from_dict(payload)
