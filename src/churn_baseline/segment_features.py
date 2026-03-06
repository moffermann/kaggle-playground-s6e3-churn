"""Fold-safe segment representation features for DA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

SEGMENT_COMPONENT_COLUMNS: Final[tuple[str, str, str]] = (
    "PaymentMethod",
    "Contract",
    "InternetService",
)
SEGMENT_KEY_COLUMN: Final[str] = "segment_key"
SEGMENT_COUNT_COLUMN: Final[str] = "segment_count"
SEGMENT_FREQ_COLUMN: Final[str] = "segment_freq"
SEGMENT_IS_RARE_COLUMN: Final[str] = "is_rare_segment"
SEGMENT_KEY_SEPARATOR: Final[str] = "__"
DEFAULT_RARE_SEGMENT_THRESHOLD: Final[float] = 0.005
DEFAULT_RARE_CATEGORY_TOKEN: Final[str] = "__RARE__"
SEGMENT_SAMPLE_WEIGHT_COLUMN: Final[str] = "segment_sample_weight"
OVERSAMPLE_POLICY_FIXED: Final[str] = "fixed"
OVERSAMPLE_POLICY_VARIANCE_TARGET: Final[str] = "variance_target"
DEFAULT_OVERSAMPLE_POLICY: Final[str] = OVERSAMPLE_POLICY_FIXED
DEFAULT_OVERSAMPLE_SE_TARGET: Final[float] = 0.0125
DEFAULT_OVERSAMPLE_SMOOTHING_ALPHA: Final[float] = 10.0


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Segment feature columns not found: {missing}")


def _build_segment_key(
    frame: pd.DataFrame,
    component_columns: Sequence[str] = SEGMENT_COMPONENT_COLUMNS,
) -> pd.Series:
    _require_columns(frame, component_columns)
    key = frame[component_columns[0]].astype(str)
    for column_name in component_columns[1:]:
        key = key + SEGMENT_KEY_SEPARATOR + frame[column_name].astype(str)
    return key


@dataclass
class SegmentRepresentationEncoder:
    """Fit/apply segment frequency features without target leakage."""

    component_columns: tuple[str, ...] = SEGMENT_COMPONENT_COLUMNS
    rare_segment_threshold: float = DEFAULT_RARE_SEGMENT_THRESHOLD
    enable_rare_bucketing: bool = False
    rare_category_token: str = DEFAULT_RARE_CATEGORY_TOKEN

    _counts_by_key: dict[str, int] | None = None
    _bucketed_component_counts: dict[str, dict[str, int]] | None = None
    _rare_values_by_component: dict[str, set[str]] | None = None
    _fit_rows: int = 0

    def fit(self, frame: pd.DataFrame) -> "SegmentRepresentationEncoder":
        """Fit segment counts on a training dataframe."""
        if self.enable_rare_bucketing:
            self._fit_component_bucketing(frame)
            key_components = self._bucket_component_frame(frame)
        else:
            key_components = frame[list(self.component_columns)].astype(str)

        keys = _build_segment_key(key_components, self.component_columns)
        counts = keys.value_counts(dropna=False)
        self._counts_by_key = {str(key): int(value) for key, value in counts.items()}
        self._fit_rows = int(len(frame))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted segment representation features to dataframe."""
        if self._counts_by_key is None or self._fit_rows <= 0:
            raise RuntimeError("SegmentRepresentationEncoder must be fitted before transform().")

        out = frame.copy()
        if self.enable_rare_bucketing:
            component_frame = self._bucket_component_frame(out)
            out = self._append_bucketing_features(out, component_frame)
        else:
            component_frame = out[list(self.component_columns)].astype(str)

        keys = _build_segment_key(component_frame, self.component_columns)
        out[SEGMENT_KEY_COLUMN] = keys

        mapped_counts = keys.map(self._counts_by_key).fillna(0).astype("int32")
        out[SEGMENT_COUNT_COLUMN] = mapped_counts
        out[SEGMENT_FREQ_COLUMN] = mapped_counts.astype("float64") / float(self._fit_rows)
        out[SEGMENT_IS_RARE_COLUMN] = (
            out[SEGMENT_FREQ_COLUMN] <= float(self.rare_segment_threshold)
        ).astype("int8")
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training dataframe."""
        return self.fit(frame).transform(frame)

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted encoder metadata."""
        if self._counts_by_key is None:
            raise RuntimeError("SegmentRepresentationEncoder is not fitted.")

        return {
            "component_columns": list(self.component_columns),
            "rare_segment_threshold": float(self.rare_segment_threshold),
            "enable_rare_bucketing": bool(self.enable_rare_bucketing),
            "rare_category_token": str(self.rare_category_token),
            "fit_rows": int(self._fit_rows),
            "counts_by_key": self._counts_by_key,
            "bucketed_component_counts": self._bucketed_component_counts,
            "rare_values_by_component": {
                key: sorted(values) for key, values in (self._rare_values_by_component or {}).items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SegmentRepresentationEncoder":
        """Construct encoder from serialized dictionary."""
        encoder = cls(
            component_columns=tuple(str(v) for v in payload["component_columns"]),
            rare_segment_threshold=float(payload["rare_segment_threshold"]),
            enable_rare_bucketing=bool(payload.get("enable_rare_bucketing", False)),
            rare_category_token=str(payload.get("rare_category_token", DEFAULT_RARE_CATEGORY_TOKEN)),
        )
        encoder._fit_rows = int(payload["fit_rows"])
        encoder._counts_by_key = {
            str(key): int(value) for key, value in dict(payload["counts_by_key"]).items()
        }
        encoder._bucketed_component_counts = {
            str(col): {str(key): int(val) for key, val in values.items()}
            for col, values in dict(payload.get("bucketed_component_counts", {})).items()
        }
        encoder._rare_values_by_component = {
            str(col): {str(value) for value in values}
            for col, values in dict(payload.get("rare_values_by_component", {})).items()
        }
        return encoder

    def _fit_component_bucketing(self, frame: pd.DataFrame) -> None:
        _require_columns(frame, self.component_columns)
        fit_rows = max(int(len(frame)), 1)
        self._bucketed_component_counts = {}
        self._rare_values_by_component = {}

        for column_name in self.component_columns:
            raw_values = frame[column_name].astype(str)
            counts = raw_values.value_counts(dropna=False)
            rare_values = {
                str(value)
                for value, count in counts.items()
                if (float(count) / float(fit_rows)) <= float(self.rare_segment_threshold)
            }
            bucketed = raw_values.where(~raw_values.isin(rare_values), self.rare_category_token)
            bucketed_counts = bucketed.value_counts(dropna=False)
            self._bucketed_component_counts[column_name] = {
                str(key): int(val) for key, val in bucketed_counts.items()
            }
            self._rare_values_by_component[column_name] = rare_values

    def _bucket_component_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.enable_rare_bucketing:
            return frame[list(self.component_columns)].astype(str)

        if self._rare_values_by_component is None:
            raise RuntimeError("Rare bucketing metadata is not fitted.")

        bucketed = pd.DataFrame(index=frame.index)
        for column_name in self.component_columns:
            raw_values = frame[column_name].astype(str)
            rare_values = self._rare_values_by_component[column_name]
            bucketed[column_name] = raw_values.where(
                ~raw_values.isin(rare_values),
                self.rare_category_token,
            )
        return bucketed

    def _append_bucketing_features(
        self,
        out: pd.DataFrame,
        bucketed_components: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._bucketed_component_counts is None:
            raise RuntimeError("Bucketed component counts are not fitted.")

        for column_name in self.component_columns:
            safe_name = column_name.lower()
            bucket_col = f"seg_{safe_name}_bucket"
            count_col = f"seg_{safe_name}_count"
            freq_col = f"seg_{safe_name}_freq"
            is_rare_col = f"seg_is_rare_{safe_name}"

            bucketed_values = bucketed_components[column_name]
            out[bucket_col] = bucketed_values
            counts_map = self._bucketed_component_counts[column_name]
            out[count_col] = bucketed_values.map(counts_map).fillna(0).astype("int32")
            out[freq_col] = out[count_col].astype("float64") / float(self._fit_rows)
            out[is_rare_col] = bucketed_values.eq(self.rare_category_token).astype("int8")
        return out


def save_segment_encoder(encoder: SegmentRepresentationEncoder, path: str | Path) -> Path:
    """Persist fitted segment encoder metadata to JSON."""
    import json

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(encoder.to_dict(), fh, indent=2)
    return out_path


def load_segment_encoder(path: str | Path) -> SegmentRepresentationEncoder:
    """Load fitted segment encoder metadata from JSON."""
    import json

    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return SegmentRepresentationEncoder.from_dict(payload)


def build_segment_sample_weights(
    segment_freq: pd.Series,
    *,
    rare_threshold: float = DEFAULT_RARE_SEGMENT_THRESHOLD,
    power: float = 0.5,
    max_weight: float = 3.0,
    normalize: bool = True,
) -> pd.Series:
    """Build per-row sample weights emphasizing rare segments."""
    if rare_threshold <= 0 or rare_threshold >= 1:
        raise ValueError("rare_threshold must be in (0, 1).")
    if power <= 0:
        raise ValueError("power must be > 0.")
    if max_weight < 1:
        raise ValueError("max_weight must be >= 1.")

    freq = pd.to_numeric(segment_freq, errors="coerce").fillna(0.0).astype("float64")
    eps = np.finfo("float64").eps
    safe_freq = freq.clip(lower=eps)
    ratio = float(rare_threshold) / safe_freq
    rare_mask = freq <= float(rare_threshold)

    weights = pd.Series(1.0, index=freq.index, dtype="float64")
    rare_weights = np.power(np.maximum(ratio[rare_mask], 1.0), float(power))
    rare_weights = np.minimum(rare_weights, float(max_weight))
    weights.loc[rare_mask] = rare_weights.astype("float64")

    if normalize:
        mean_weight = float(weights.mean())
        if mean_weight > 0:
            weights = weights / mean_weight
    weights.name = SEGMENT_SAMPLE_WEIGHT_COLUMN
    return weights


def _scale_allocation_to_budget(
    allocation: dict[str, int],
    budget: int,
) -> dict[str, int]:
    """Scale non-negative integer allocation to fit a global budget."""
    if budget < 0:
        raise ValueError("budget must be >= 0")
    if not allocation:
        return {}

    total_requested = int(sum(max(int(v), 0) for v in allocation.values()))
    if total_requested <= budget:
        return {str(k): int(max(int(v), 0)) for k, v in allocation.items()}
    if budget == 0:
        return {str(k): 0 for k in allocation}

    scale = float(budget) / float(total_requested)
    scaled_float: dict[str, float] = {
        str(k): float(max(int(v), 0)) * scale for k, v in allocation.items()
    }
    scaled_floor: dict[str, int] = {k: int(np.floor(v)) for k, v in scaled_float.items()}
    remaining = int(budget - sum(scaled_floor.values()))

    if remaining > 0:
        remainders = sorted(
            ((k, scaled_float[k] - float(scaled_floor[k])) for k in scaled_float),
            key=lambda item: (-item[1], item[0]),
        )
        for key, _ in remainders[:remaining]:
            scaled_floor[key] += 1

    return scaled_floor


def oversample_rare_segments(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    segment_key_column: str = SEGMENT_KEY_COLUMN,
    segment_count_column: str = SEGMENT_COUNT_COLUMN,
    target_min_freq: float = DEFAULT_RARE_SEGMENT_THRESHOLD,
    max_multiplier: float = 2.0,
    max_added_rate: float = 0.03,
    policy: str = DEFAULT_OVERSAMPLE_POLICY,
    se_target: float = DEFAULT_OVERSAMPLE_SE_TARGET,
    smoothing_alpha: float = DEFAULT_OVERSAMPLE_SMOOTHING_ALPHA,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float | int]]:
    """Conservative in-fold oversampling for rare segment keys."""
    if target_min_freq <= 0 or target_min_freq >= 1:
        raise ValueError("target_min_freq must be in (0, 1).")
    if max_multiplier < 1:
        raise ValueError("max_multiplier must be >= 1.")
    if max_added_rate < 0:
        raise ValueError("max_added_rate must be >= 0.")
    if policy not in {OVERSAMPLE_POLICY_FIXED, OVERSAMPLE_POLICY_VARIANCE_TARGET}:
        raise ValueError(
            "policy must be one of: fixed, variance_target"
        )
    if se_target <= 0:
        raise ValueError("se_target must be > 0.")
    if smoothing_alpha < 0:
        raise ValueError("smoothing_alpha must be >= 0.")

    _require_columns(features, [segment_key_column, segment_count_column])
    if len(features) != len(target):
        raise ValueError("features and target must have same number of rows.")

    train_rows = int(len(features))
    if train_rows == 0:
        return (
            features.copy(),
            target.copy(),
            {
                "added_rows": 0,
                "added_rate": 0.0,
                "requested_added_rows": 0,
                "budget_scaling_applied": False,
                "target_min_count": 0,
                "global_added_cap": 0,
                "rare_segments_before": 0,
                "augmented_segments": 0,
                "policy": str(policy),
                "se_target": float(se_target),
                "smoothing_alpha": float(smoothing_alpha),
            },
        )

    target_min_count = max(int(np.ceil(float(target_min_freq) * float(train_rows))), 1)
    global_added_cap = int(np.floor(float(max_added_rate) * float(train_rows)))
    if global_added_cap <= 0 or max_multiplier == 1:
        rare_segment_count = int(
            (features[segment_key_column].astype(str).value_counts(dropna=False) < target_min_count).sum()
        )
        return (
            features.copy(),
            target.copy(),
            {
                "added_rows": 0,
                "added_rate": 0.0,
                "requested_added_rows": 0,
                "budget_scaling_applied": False,
                "target_min_count": int(target_min_count),
                "global_added_cap": int(global_added_cap),
                "rare_segments_before": rare_segment_count,
                "augmented_segments": 0,
                "policy": str(policy),
                "se_target": float(se_target),
                "smoothing_alpha": float(smoothing_alpha),
            },
        )

    key_series = features[segment_key_column].astype(str)
    counts_by_key = key_series.value_counts(dropna=False)
    rare_keys = counts_by_key[counts_by_key < target_min_count]

    if rare_keys.empty:
        return (
            features.copy(),
            target.copy(),
            {
                "added_rows": 0,
                "added_rate": 0.0,
                "requested_added_rows": 0,
                "budget_scaling_applied": False,
                "target_min_count": int(target_min_count),
                "global_added_cap": int(global_added_cap),
                "rare_segments_before": 0,
                "augmented_segments": 0,
                "policy": str(policy),
                "se_target": float(se_target),
                "smoothing_alpha": float(smoothing_alpha),
            },
        )

    rng = np.random.default_rng(int(random_state))
    global_target_rate = float(pd.to_numeric(target, errors="coerce").astype("float64").mean())
    global_target_rate = float(np.clip(global_target_rate, 1e-6, 1.0 - 1e-6))
    key_target_df = pd.DataFrame(
        {
            "segment_key": key_series.values,
            "target": pd.to_numeric(target, errors="coerce").astype("float64").values,
        }
    )
    target_sum_by_key = key_target_df.groupby("segment_key", dropna=False)["target"].sum()

    requested_added_by_key: dict[str, int] = {}

    for segment_key, segment_count_raw in rare_keys.items():
        segment_count = int(segment_count_raw)
        if segment_count <= 0:
            continue

        if policy == OVERSAMPLE_POLICY_FIXED:
            target_count_policy = int(target_min_count)
        else:
            target_sum = float(target_sum_by_key.get(str(segment_key), 0.0))
            p_hat = (target_sum + float(smoothing_alpha) * global_target_rate) / (
                float(segment_count) + float(smoothing_alpha)
            )
            p_hat = float(np.clip(p_hat, 1e-6, 1.0 - 1e-6))
            target_count_policy = int(np.ceil((p_hat * (1.0 - p_hat)) / (float(se_target) ** 2)))

        add_needed = max(int(target_count_policy - segment_count), 0)
        add_cap_multiplier = int(np.floor(float(segment_count) * (float(max_multiplier) - 1.0)))
        add_rows = min(add_needed, add_cap_multiplier, global_added_cap)
        if add_rows <= 0:
            continue

        requested_added_by_key[str(segment_key)] = int(add_rows)

    if not requested_added_by_key:
        return (
            features.copy(),
            target.copy(),
            {
                "added_rows": 0,
                "added_rate": 0.0,
                "requested_added_rows": 0,
                "budget_scaling_applied": False,
                "target_min_count": int(target_min_count),
                "global_added_cap": int(global_added_cap),
                "rare_segments_before": int(len(rare_keys)),
                "augmented_segments": 0,
                "policy": str(policy),
                "se_target": float(se_target),
                "smoothing_alpha": float(smoothing_alpha),
            },
        )

    final_added_by_key = _scale_allocation_to_budget(requested_added_by_key, global_added_cap)
    sampled_indices: list[np.ndarray] = []
    augmented_segments = 0
    for segment_key, add_rows in final_added_by_key.items():
        if add_rows <= 0:
            continue
        member_idx = key_series[key_series == str(segment_key)].index.to_numpy()
        sampled = rng.choice(member_idx, size=int(add_rows), replace=True)
        sampled_indices.append(sampled)
        augmented_segments += 1

    if not sampled_indices:
        return (
            features.copy(),
            target.copy(),
            {
                "added_rows": 0,
                "added_rate": 0.0,
                "requested_added_rows": int(sum(requested_added_by_key.values())),
                "budget_scaling_applied": False,
                "target_min_count": int(target_min_count),
                "global_added_cap": int(global_added_cap),
                "rare_segments_before": int(len(rare_keys)),
                "augmented_segments": 0,
                "policy": str(policy),
                "se_target": float(se_target),
                "smoothing_alpha": float(smoothing_alpha),
            },
        )

    all_added_idx = np.concatenate(sampled_indices)

    x_aug = pd.concat([features, features.loc[all_added_idx]], axis=0, ignore_index=True)
    y_aug = pd.concat([target, target.loc[all_added_idx]], axis=0, ignore_index=True)

    requested_added_rows = int(sum(requested_added_by_key.values()))
    final_added_rows = int(len(all_added_idx))
    summary = {
        "added_rows": final_added_rows,
        "added_rate": float(final_added_rows / float(train_rows)),
        "requested_added_rows": requested_added_rows,
        "budget_scaling_applied": bool(requested_added_rows > final_added_rows),
        "target_min_count": int(target_min_count),
        "global_added_cap": int(global_added_cap),
        "rare_segments_before": int(len(rare_keys)),
        "augmented_segments": int(augmented_segments),
        "policy": str(policy),
        "se_target": float(se_target),
        "smoothing_alpha": float(smoothing_alpha),
        "rows_before": int(train_rows),
        "rows_after": int(len(x_aug)),
    }
    return x_aug, y_aug, summary
