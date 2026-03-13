"""Counterfactual teacher-sensitivity smoke against incumbent v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import load_csv
from .diagnostics import build_family_frame, load_reference_prediction_frame
from .evaluation import binary_auc
from .feature_engineering import apply_feature_engineering, normalize_feature_blocks, partition_feature_blocks
from .modeling import load_model, predict_proba
from .uncertainty_band import DEFAULT_V3_OOF_SPEC

DEFAULT_COMPONENT_WEIGHTS_JSON = "artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json"
SUPPORTED_COMPONENTS: tuple[str, ...] = ("cb", "r", "rv")
SUPPORTED_COUNTERFACTUALS: tuple[str, ...] = (
    "auto_payment",
    "paperless_off",
    "contract_upgrade",
    "stability_bundle",
)
SUPPORTED_SIGNALS: tuple[str, ...] = (
    "stable_bundle_drop",
    "mean_positive_drop",
    "max_positive_drop",
)
MIN_MASK_ROWS = 2000


def _clip_probability(values: np.ndarray | pd.Series, epsilon: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(values, dtype="float64"), epsilon, 1.0 - epsilon)


def _load_v3_reference_frame(
    *,
    reference_v3_oof: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return load_reference_prediction_frame(
        reference_oof_spec=reference_v3_oof,
        id_column=ID_COLUMN,
        target_column="target",
    )


def _align_reference_prediction(
    train_df: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    required = {ID_COLUMN, "target", "reference_pred"}
    missing = sorted(required.difference(reference_frame.columns))
    if missing:
        raise ValueError(f"reference_frame missing required columns: {missing}")

    merged = train_df[[ID_COLUMN, TARGET_COLUMN]].merge(
        reference_frame[[ID_COLUMN, "target", "reference_pred"]],
        how="left",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if merged["reference_pred"].isna().any():
        sample_ids = merged.loc[merged["reference_pred"].isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference_frame missing ids from train.csv: {sample_ids}")

    train_target = (train_df[TARGET_COLUMN].astype(str).str.lower() == "yes").astype("int8")
    reference_target = pd.to_numeric(merged["target"], errors="raise").astype("int8")
    if not np.array_equal(train_target.to_numpy(dtype="int8"), reference_target.to_numpy(dtype="int8")):
        raise ValueError("reference_frame target does not align with train.csv")

    reference_pred = pd.Series(
        pd.to_numeric(merged["reference_pred"], errors="raise").astype("float64").values,
        index=train_df.index,
        dtype="float64",
        name="reference_pred",
    )
    return train_target, reference_pred


def _load_component_weights(
    weights_json_path: str | Path,
    *,
    allowed_components: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads(Path(weights_json_path).read_text(encoding="utf-8"))
    weights_raw = payload.get("weights")
    if not isinstance(weights_raw, dict) or not weights_raw:
        raise ValueError(f"weights json {weights_json_path} does not contain a valid 'weights' object")

    selected = {
        str(name): float(value)
        for name, value in weights_raw.items()
        if str(name) in set(allowed_components)
    }
    if not selected:
        raise ValueError(
            f"weights json {weights_json_path} does not contain any of the required components {list(allowed_components)}"
        )

    weight_sum = float(sum(selected.values()))
    if weight_sum <= 0.0:
        raise ValueError(f"selected component weights must sum to > 0, got {weight_sum}")

    normalized = {name: float(value / weight_sum) for name, value in selected.items()}
    component_paths = payload.get("components", {})
    if not isinstance(component_paths, dict):
        raise ValueError(f"weights json {weights_json_path} must contain a 'components' mapping")
    return normalized, component_paths


def _prepare_feature_matrix(raw_frame: pd.DataFrame, feature_blocks: Sequence[str] | None) -> pd.DataFrame:
    normalized_blocks = normalize_feature_blocks(feature_blocks)
    stateless_blocks, stateful_blocks = partition_feature_blocks(normalized_blocks)
    if stateful_blocks:
        raise ValueError(
            "counterfactual sensitivity smoke only supports stateless blocks. "
            f"Got stateful blocks: {list(stateful_blocks)}"
        )

    features = raw_frame.copy()
    drop_columns = [column for column in (ID_COLUMN, TARGET_COLUMN) if column in features.columns]
    if drop_columns:
        features = features.drop(columns=drop_columns)
    return apply_feature_engineering(features, feature_blocks=stateless_blocks)


def _apply_counterfactual(raw_frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = raw_frame.copy()
    payment = out["PaymentMethod"].astype(str)
    paperless = out["PaperlessBilling"].astype(str)
    contract = out["Contract"].astype(str)

    if name == "auto_payment":
        mapped = payment.replace(
            {
                "Electronic check": "Bank transfer (automatic)",
                "Mailed check": "Credit card (automatic)",
            }
        )
        out["PaymentMethod"] = mapped
        return out

    if name == "paperless_off":
        out["PaperlessBilling"] = np.where(paperless.eq("Yes"), "No", paperless)
        return out

    if name == "contract_upgrade":
        mapped = contract.replace(
            {
                "Month-to-month": "One year",
                "One year": "Two year",
            }
        )
        out["Contract"] = mapped
        return out

    if name == "stability_bundle":
        out = _apply_counterfactual(out, "auto_payment")
        out = _apply_counterfactual(out, "paperless_off")
        out = _apply_counterfactual(out, "contract_upgrade")
        return out

    raise ValueError(f"Unsupported counterfactual '{name}'. Available: {list(SUPPORTED_COUNTERFACTUALS)}")


def _predict_catboost_average(
    feature_frame: pd.DataFrame,
    model_paths: Sequence[str | Path],
) -> np.ndarray:
    if not model_paths:
        raise ValueError("model_paths must not be empty")

    prediction_rows: list[np.ndarray] = []
    for model_path in model_paths:
        model = load_model(model_path)
        prediction_rows.append(predict_proba(model, feature_frame).to_numpy(dtype="float64"))
    return np.mean(np.column_stack(prediction_rows), axis=1)


def _resolve_component_model_paths(
    component_paths: Mapping[str, Any],
    *,
    component: str,
) -> list[str]:
    if component == "cb":
        raw_paths = component_paths.get("cb_models")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("weights json does not contain a valid 'cb_models' list")
        return [str(path) for path in raw_paths]
    if component == "r":
        model_path = component_paths.get("r_model")
        if not model_path:
            raise ValueError("weights json does not contain 'r_model'")
        return [str(model_path)]
    if component == "rv":
        model_path = component_paths.get("rv_model")
        if not model_path:
            raise ValueError("weights json does not contain 'rv_model'")
        return [str(model_path)]
    raise ValueError(f"Unsupported component '{component}'. Available: {list(SUPPORTED_COMPONENTS)}")


def _component_feature_blocks(component: str) -> tuple[str, ...]:
    if component == "cb":
        return ()
    if component == "r":
        return ("R",)
    if component == "rv":
        return ("R", "V")
    raise ValueError(f"Unsupported component '{component}'. Available: {list(SUPPORTED_COMPONENTS)}")


def _predict_component_scenarios(
    raw_focus_frame: pd.DataFrame,
    *,
    component: str,
    model_paths: Sequence[str | Path],
    counterfactuals: Sequence[str],
) -> dict[str, np.ndarray]:
    scenario_frames: dict[str, pd.DataFrame] = {"base": raw_focus_frame}
    for name in counterfactuals:
        scenario_frames[name] = _apply_counterfactual(raw_focus_frame, name)

    predictions: dict[str, np.ndarray] = {}
    feature_blocks = _component_feature_blocks(component)
    for scenario_name, scenario_frame in scenario_frames.items():
        feature_frame = _prepare_feature_matrix(scenario_frame, feature_blocks)
        predictions[scenario_name] = _predict_catboost_average(feature_frame, model_paths)
    return predictions


def _build_counterfactual_signals(
    raw_focus_frame: pd.DataFrame,
    *,
    component_weights: Mapping[str, float],
    component_paths: Mapping[str, Any],
    counterfactuals: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weighted_base = np.zeros(len(raw_focus_frame), dtype="float64")
    weighted_counterfactuals = {
        name: np.zeros(len(raw_focus_frame), dtype="float64")
        for name in counterfactuals
    }
    component_rows: list[dict[str, Any]] = []

    for component, weight in component_weights.items():
        model_paths = _resolve_component_model_paths(component_paths, component=component)
        scenario_predictions = _predict_component_scenarios(
            raw_focus_frame,
            component=component,
            model_paths=model_paths,
            counterfactuals=counterfactuals,
        )
        weighted_base += float(weight) * scenario_predictions["base"]
        component_rows.append(
            {
                "component": str(component),
                "weight": float(weight),
                "model_count": int(len(model_paths)),
                "base_pred_mean": float(np.mean(scenario_predictions["base"])),
                "base_pred_std": float(np.std(scenario_predictions["base"], ddof=0)),
            }
        )
        for name in counterfactuals:
            weighted_counterfactuals[name] += float(weight) * scenario_predictions[name]
            component_rows[-1][f"{name}_delta_mean"] = float(
                np.mean(scenario_predictions[name] - scenario_predictions["base"])
            )

    signal_frame = pd.DataFrame(index=raw_focus_frame.index)
    signal_frame["cf_base_teacher_pred"] = _clip_probability(weighted_base)

    positive_drop_columns: list[str] = []
    for name in counterfactuals:
        scenario_pred = _clip_probability(weighted_counterfactuals[name])
        signal_frame[f"cf_{name}_pred"] = scenario_pred
        signal_frame[f"cf_{name}_delta"] = scenario_pred - signal_frame["cf_base_teacher_pred"].to_numpy(dtype="float64")
        signal_frame[f"cf_{name}_positive_drop"] = np.clip(
            signal_frame["cf_base_teacher_pred"].to_numpy(dtype="float64") - scenario_pred,
            a_min=0.0,
            a_max=None,
        )
        positive_drop_columns.append(f"cf_{name}_positive_drop")

    positive_drop_matrix = signal_frame[positive_drop_columns].to_numpy(dtype="float64")
    signal_frame["cf_mean_positive_drop"] = positive_drop_matrix.mean(axis=1)
    signal_frame["cf_max_positive_drop"] = positive_drop_matrix.max(axis=1)
    signal_frame["cf_stable_bundle_drop"] = signal_frame["cf_stability_bundle_positive_drop"].astype("float64")

    summary = {
        "component_rows": component_rows,
        "counterfactuals": list(counterfactuals),
        "signal_summary": {
            "mean_positive_drop_mean": float(signal_frame["cf_mean_positive_drop"].mean()),
            "mean_positive_drop_std": float(signal_frame["cf_mean_positive_drop"].std(ddof=0)),
            "max_positive_drop_mean": float(signal_frame["cf_max_positive_drop"].mean()),
            "stable_bundle_drop_mean": float(signal_frame["cf_stable_bundle_drop"].mean()),
        },
    }
    return signal_frame, summary


def _build_focus_mask(
    train_df: pd.DataFrame,
    *,
    reference_pred: pd.Series,
    target_family_level: str,
    target_family_value: str,
    reference_band_half_width: float | None,
) -> tuple[pd.Series, dict[str, Any]]:
    if target_family_level not in {"segment3", "segment5"}:
        raise ValueError("target_family_level must be 'segment3' or 'segment5'")
    if reference_band_half_width is not None and (reference_band_half_width <= 0.0 or reference_band_half_width >= 0.5):
        raise ValueError(f"reference_band_half_width must be in (0, 0.5), got {reference_band_half_width}")

    family_frame = build_family_frame(train_df)
    family_mask = family_frame[target_family_level].astype(str).eq(str(target_family_value))
    if reference_band_half_width is None:
        final_mask = family_mask
    else:
        band_mask = reference_pred.sub(0.5).abs().le(float(reference_band_half_width))
        final_mask = family_mask & band_mask

    final_rows = int(final_mask.sum())
    if final_rows < MIN_MASK_ROWS:
        raise ValueError(
            f"counterfactual focus mask selects only {final_rows} rows. Minimum required: {MIN_MASK_ROWS}"
        )

    summary = {
        "target_family_level": str(target_family_level),
        "target_family_value": str(target_family_value),
        "reference_band_half_width": None if reference_band_half_width is None else float(reference_band_half_width),
        "family_rows": int(family_mask.sum()),
        "mask_rows": final_rows,
        "mask_rate": float(final_rows / max(len(train_df), 1)),
        "reference_mean_on_mask": float(reference_pred.loc[final_mask].mean()),
        "reference_std_on_mask": float(reference_pred.loc[final_mask].std(ddof=0)),
    }
    return final_mask.astype(bool), summary


def _parse_signal_value(signal_frame: pd.DataFrame, signal_name: str) -> pd.Series:
    column_map = {
        "stable_bundle_drop": "cf_stable_bundle_drop",
        "mean_positive_drop": "cf_mean_positive_drop",
        "max_positive_drop": "cf_max_positive_drop",
    }
    if signal_name not in column_map:
        raise ValueError(f"Unsupported signal_name '{signal_name}'. Available: {list(SUPPORTED_SIGNALS)}")
    return signal_frame[column_map[signal_name]].astype("float64")


def _scan_counterfactual_candidate(
    y: pd.Series,
    reference_pred: pd.Series,
    mask: pd.Series,
    signal_frame: pd.DataFrame,
    *,
    signal_names: Sequence[str],
    alpha_grid: Sequence[float],
) -> tuple[pd.Series, dict[str, Any], list[dict[str, Any]]]:
    if not signal_names:
        raise ValueError("signal_names must not be empty")
    if not alpha_grid:
        raise ValueError("alpha_grid must not be empty")

    best_auc = float(binary_auc(y, reference_pred))
    best_pred = reference_pred.copy()
    best_row = {
        "signal_name": "reference",
        "alpha": 0.0,
        "candidate_oof_auc": best_auc,
        "delta_vs_reference_oof_auc": 0.0,
        "mask_rows": int(mask.sum()),
    }
    scan_rows: list[dict[str, Any]] = [dict(best_row)]

    for signal_name in signal_names:
        signal = _parse_signal_value(signal_frame, signal_name)
        for alpha in alpha_grid:
            candidate = reference_pred.copy()
            candidate.loc[mask] = _clip_probability(
                reference_pred.loc[mask].to_numpy(dtype="float64") + float(alpha) * signal.loc[mask].to_numpy(dtype="float64")
            )
            auc = float(binary_auc(y, candidate))
            row = {
                "signal_name": str(signal_name),
                "alpha": float(alpha),
                "candidate_oof_auc": auc,
                "delta_vs_reference_oof_auc": float(auc - best_auc),
                "mask_rows": int(mask.sum()),
                "signal_mean_on_mask": float(signal.loc[mask].mean()),
                "signal_std_on_mask": float(signal.loc[mask].std(ddof=0)),
            }
            scan_rows.append(row)
            if auc > best_row["candidate_oof_auc"] + 1e-15:
                best_row = dict(row)
                best_pred = candidate
    best_row["reference_oof_auc"] = best_auc
    best_row["delta_vs_reference_oof_auc"] = float(best_row["candidate_oof_auc"] - best_auc)
    return best_pred, best_row, scan_rows


def run_counterfactual_sensitivity_smoke(
    *,
    train_csv_path: str | Path,
    reference_v3_oof: str,
    component_weights_json: str | Path,
    target_family_level: str,
    target_family_value: str,
    reference_band_half_width: float | None,
    alpha_grid: Sequence[float],
    counterfactuals: Sequence[str],
    signal_names: Sequence[str],
    metrics_path: str | Path,
    oof_path: str | Path,
) -> dict[str, Any]:
    train_df = load_csv(train_csv_path)
    target, reference_pred = _align_reference_prediction(
        train_df,
        _load_v3_reference_frame(reference_v3_oof=reference_v3_oof)[0],
    )
    focus_mask, mask_summary = _build_focus_mask(
        train_df,
        reference_pred=reference_pred,
        target_family_level=target_family_level,
        target_family_value=target_family_value,
        reference_band_half_width=reference_band_half_width,
    )

    raw_focus_frame = train_df.loc[focus_mask].copy()
    component_weights, component_paths = _load_component_weights(
        component_weights_json,
        allowed_components=SUPPORTED_COMPONENTS,
    )
    signal_frame, signal_summary = _build_counterfactual_signals(
        raw_focus_frame,
        component_weights=component_weights,
        component_paths=component_paths,
        counterfactuals=counterfactuals,
    )

    full_signal_frame = pd.DataFrame(index=train_df.index)
    full_signal_frame["cf_mean_positive_drop"] = 0.0
    full_signal_frame["cf_max_positive_drop"] = 0.0
    full_signal_frame["cf_stable_bundle_drop"] = 0.0
    for column in signal_frame.columns:
        full_signal_frame[column] = 0.0
        full_signal_frame.loc[focus_mask, column] = signal_frame[column].to_numpy(dtype="float64")

    candidate_pred, best_row, scan_rows = _scan_counterfactual_candidate(
        target,
        reference_pred,
        focus_mask,
        full_signal_frame,
        signal_names=signal_names,
        alpha_grid=alpha_grid,
    )

    analysis_oof = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].to_numpy(),
            "target": target.to_numpy(dtype="int8"),
            "reference_pred": reference_pred.to_numpy(dtype="float64"),
            "candidate_pred": candidate_pred.to_numpy(dtype="float64"),
        }
    )
    ensure_parent_dir(oof_path)
    analysis_oof.to_csv(oof_path, index=False)

    metrics = {
        "experiment_name": "counterfactual_teacher_sensitivity",
        "reference_v3_oof": str(reference_v3_oof),
        "component_weights_json": str(component_weights_json),
        "component_weights_subset": {name: float(value) for name, value in component_weights.items()},
        "target_family_level": str(target_family_level),
        "target_family_value": str(target_family_value),
        "reference_band_half_width": None if reference_band_half_width is None else float(reference_band_half_width),
        "counterfactuals": [str(name) for name in counterfactuals],
        "signal_names": [str(name) for name in signal_names],
        "mask_summary": mask_summary,
        "signal_summary": signal_summary,
        "best_signal_name": str(best_row["signal_name"]),
        "best_alpha": float(best_row["alpha"]),
        "reference_oof_auc": float(best_row["reference_oof_auc"]),
        "candidate_oof_auc": float(best_row["candidate_oof_auc"]),
        "delta_vs_reference_oof_auc": float(best_row["delta_vs_reference_oof_auc"]),
        "scan_rows": scan_rows,
        "oof_path": str(oof_path),
        "metrics_path": str(metrics_path),
    }
    write_json(metrics_path, metrics)
    return metrics
