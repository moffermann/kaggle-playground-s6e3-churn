"""Offline family-level pseudo-labeling experiments."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .artifacts import ensure_parent_dir, write_json
from .config import CatBoostHyperParams, ID_COLUMN
from .data import infer_categorical_columns, load_csv, prepare_train_features
from .evaluation import binary_auc
from .feature_engineering import (
    BLOCK_G,
    apply_coverage_backoff_features,
    fit_coverage_backoff_state,
    normalize_feature_blocks,
    partition_feature_blocks,
)
from .modeling import best_iteration_or_default, fit_with_validation, predict_proba


DEFAULT_FAMILY_KEY = "Electronic check__Month-to-month__Fiber optic__Yes__13_24"
_MIN_CLASS_COUNT = 2
LABEL_MODE_HARD = "hard"
LABEL_MODE_SOFT = "soft"
SUPPORTED_LABEL_MODES = (LABEL_MODE_HARD, LABEL_MODE_SOFT)


def _build_tenure_bin(tenure_values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(tenure_values, errors="coerce").fillna(0.0),
        bins=[-np.inf, 6, 12, 24, 48, np.inf],
        labels=["0_6", "7_12", "13_24", "25_48", "49_plus"],
    ).astype(str)


def build_segment5_family(frame: pd.DataFrame) -> pd.Series:
    """Build segment5 family key from the raw train/test schema."""
    required = ("PaymentMethod", "Contract", "InternetService", "PaperlessBilling", "tenure")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"segment5 family requires columns: {missing}")

    tenure_bin = _build_tenure_bin(frame["tenure"])
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


def list_supported_label_modes() -> tuple[str, ...]:
    """Return supported pseudo-label target modes."""
    return SUPPORTED_LABEL_MODES


def _build_split_stratify_labels(
    y: pd.Series,
    family_mask: pd.Series,
    *,
    min_count: int = _MIN_CLASS_COUNT,
) -> pd.Series:
    base = y.astype(str)
    joint = base + "__" + family_mask.astype("int8").astype(str)
    counts = joint.value_counts(dropna=False)
    if not counts.empty and int(counts.min()) >= int(min_count):
        return joint
    return base


def _resolve_component_frame(
    train_df: pd.DataFrame,
    reference_component_frame: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if reference_component_frame is None:
        return None
    if ID_COLUMN not in reference_component_frame.columns:
        raise ValueError(f"reference_component_frame must contain '{ID_COLUMN}'")
    component_columns = [
        column for column in reference_component_frame.columns if column != ID_COLUMN and str(column).startswith("pred_")
    ]
    if not component_columns:
        raise ValueError("reference_component_frame must contain at least one 'pred_*' column")
    aligned = train_df[[ID_COLUMN]].merge(
        reference_component_frame[[ID_COLUMN, *component_columns]].copy(),
        how="left",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if aligned[component_columns].isna().any().any():
        missing_ids = aligned.loc[aligned[component_columns].isna().any(axis=1), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference_component_frame missing train ids: {missing_ids}")
    return aligned


def _apply_stateful_views(
    x_fit: pd.DataFrame,
    stateful_blocks: Sequence[str],
    apply_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    fit_out = x_fit.copy()
    outputs = {name: frame.copy() for name, frame in apply_frames.items()}
    if not stateful_blocks:
        return fit_out, outputs

    normalized_stateful = tuple(stateful_blocks)
    if BLOCK_G in normalized_stateful:
        coverage_state = fit_coverage_backoff_state(fit_out)
        fit_out = apply_coverage_backoff_features(fit_out, coverage_state)
        outputs = {
            name: apply_coverage_backoff_features(frame, coverage_state) for name, frame in outputs.items()
        }
    unsupported = [block for block in normalized_stateful if block != BLOCK_G]
    if unsupported:
        raise ValueError(
            "Unsupported stateful feature blocks for pseudo-labeling experiment: "
            f"{unsupported}. Update _apply_stateful_views before enabling them here."
        )
    return fit_out, outputs


def _safe_auc_on_mask(y_true: pd.Series, pred: pd.Series, mask: pd.Series) -> float | None:
    if int(mask.sum()) < 2:
        return None
    masked_y = y_true.loc[mask]
    if masked_y.nunique(dropna=False) < 2:
        return None
    return float(binary_auc(masked_y, pred.loc[mask]))


def _teacher_std(component_frame: pd.DataFrame | None, index: pd.Index) -> pd.Series | None:
    if component_frame is None:
        return None
    component_columns = [column for column in component_frame.columns if column != ID_COLUMN]
    std = component_frame.loc[index, component_columns].astype("float64").std(axis=1, ddof=0)
    return pd.Series(std.values, index=index, dtype="float64")


def _teacher_agreement_mask(
    component_frame: pd.DataFrame | None,
    index: pd.Index,
    pseudo_target: pd.Series,
) -> pd.Series:
    if component_frame is None:
        raise ValueError("teacher agreement requires reference_component_frame")
    component_columns = [column for column in component_frame.columns if column != ID_COLUMN]
    teacher_matrix = component_frame.loc[index, component_columns].astype("float64")
    agree_positive = teacher_matrix.ge(0.5).all(axis=1)
    agree_negative = teacher_matrix.lt(0.5).all(axis=1)
    return pd.Series(
        np.where(pseudo_target.astype(int).eq(1), agree_positive, agree_negative),
        index=index,
        dtype="bool",
    )


def _select_pseudo_rows(
    *,
    y_true: pd.Series,
    reference_pred: pd.Series,
    component_frame: pd.DataFrame | None,
    upper_threshold: float,
    lower_threshold: float,
    require_teacher_agreement: bool,
    max_teacher_std: float | None,
    min_selected_rows: int,
) -> tuple[pd.Index, pd.Series, pd.Series, dict[str, Any]]:
    if not (0.5 < upper_threshold <= 1.0):
        raise ValueError(f"upper_threshold must be in (0.5, 1.0], got {upper_threshold}")
    if not (0.0 <= lower_threshold < 0.5):
        raise ValueError(f"lower_threshold must be in [0.0, 0.5), got {lower_threshold}")

    positive_mask = reference_pred.ge(float(upper_threshold))
    negative_mask = reference_pred.le(float(lower_threshold))
    selected_mask = positive_mask | negative_mask
    selected_index = reference_pred.index[selected_mask]
    if len(selected_index) == 0:
        return selected_index, pd.Series(dtype="int8"), pd.Series(dtype="float64"), {
            "selected_reference_mean": None,
            "selected_rows": 0,
            "selected_positive_rows": 0,
            "selected_negative_rows": 0,
            "selected_accuracy": None,
            "selected_positive_precision": None,
            "selected_negative_precision": None,
            "selected_mean_confidence": None,
            "teacher_std_mean": None,
        }

    pseudo_target = pd.Series(
        np.where(positive_mask.loc[selected_index], 1, 0),
        index=selected_index,
        dtype="int8",
    )

    if require_teacher_agreement:
        agreement_mask = _teacher_agreement_mask(component_frame, selected_index, pseudo_target)
        selected_index = agreement_mask.index[agreement_mask]
        pseudo_target = pseudo_target.loc[selected_index]

    teacher_std = _teacher_std(component_frame, selected_index)
    if max_teacher_std is not None and teacher_std is not None:
        std_mask = teacher_std.le(float(max_teacher_std))
        selected_index = std_mask.index[std_mask]
        pseudo_target = pseudo_target.loc[selected_index]
        teacher_std = teacher_std.loc[selected_index]

    if len(selected_index) < int(min_selected_rows):
        return selected_index, pseudo_target, reference_pred.loc[selected_index].astype("float64"), {
            "selected_reference_mean": (
                None if len(selected_index) == 0 else float(reference_pred.loc[selected_index].astype("float64").mean())
            ),
            "selected_rows": int(len(selected_index)),
            "selected_positive_rows": int(pseudo_target.sum()),
            "selected_negative_rows": int(len(selected_index) - int(pseudo_target.sum())),
            "selected_accuracy": None,
            "selected_positive_precision": None,
            "selected_negative_precision": None,
            "selected_mean_confidence": None,
            "teacher_std_mean": None if teacher_std is None or teacher_std.empty else float(teacher_std.mean()),
        }

    selected_truth = y_true.loc[selected_index].astype(int)
    selected_reference = reference_pred.loc[selected_index].astype("float64")
    selected_positive = pseudo_target.eq(1)
    selected_negative = ~selected_positive
    positive_precision = (
        float((selected_truth.loc[selected_positive] == 1).mean())
        if int(selected_positive.sum()) > 0
        else None
    )
    negative_precision = (
        float((selected_truth.loc[selected_negative] == 0).mean())
        if int(selected_negative.sum()) > 0
        else None
    )
    metrics = {
        "selected_reference_mean": float(selected_reference.mean()),
        "selected_rows": int(len(selected_index)),
        "selected_positive_rows": int(selected_positive.sum()),
        "selected_negative_rows": int(selected_negative.sum()),
        "selected_accuracy": float((selected_truth == pseudo_target).mean()),
        "selected_positive_precision": positive_precision,
        "selected_negative_precision": negative_precision,
        "selected_mean_confidence": float(np.maximum(selected_reference, 1.0 - selected_reference).mean()),
        "teacher_std_mean": None if teacher_std is None or teacher_std.empty else float(teacher_std.mean()),
    }
    return selected_index, pseudo_target, selected_reference, metrics


def _build_pseudo_training_rows(
    *,
    x_selected: pd.DataFrame,
    pseudo_target: pd.Series,
    selected_reference: pd.Series,
    pseudo_weight: float,
    label_mode: str,
    scale_weight_by_confidence: bool,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    label_mode_value = str(label_mode).strip().lower()
    if label_mode_value not in SUPPORTED_LABEL_MODES:
        raise ValueError(f"Unsupported label_mode '{label_mode}'. Supported: {SUPPORTED_LABEL_MODES}")

    base_confidence = np.ones(len(x_selected), dtype="float64")
    if scale_weight_by_confidence:
        base_confidence = np.clip(2.0 * np.abs(selected_reference.to_numpy(dtype="float64") - 0.5), 0.0, 1.0)

    if label_mode_value == LABEL_MODE_HARD:
        weights = np.asarray(base_confidence, dtype="float64") * float(pseudo_weight)
        return (
            x_selected.copy(),
            pseudo_target.astype("int8").copy(),
            weights,
        )

    positive_rows = x_selected.copy()
    positive_target = pd.Series(np.ones(len(x_selected), dtype="int8"), index=x_selected.index)
    positive_weight = (
        selected_reference.to_numpy(dtype="float64") * np.asarray(base_confidence, dtype="float64") * float(pseudo_weight)
    )

    negative_rows = x_selected.copy()
    negative_rows.index = pd.Index([f"{idx}__soft_neg" for idx in x_selected.index], dtype="object")
    negative_target = pd.Series(np.zeros(len(x_selected), dtype="int8"), index=negative_rows.index)
    negative_weight = (
        (1.0 - selected_reference.to_numpy(dtype="float64"))
        * np.asarray(base_confidence, dtype="float64")
        * float(pseudo_weight)
    )

    positive_rows.index = pd.Index([f"{idx}__soft_pos" for idx in x_selected.index], dtype="object")
    positive_target.index = positive_rows.index
    pseudo_x = pd.concat([positive_rows, negative_rows], axis=0)
    pseudo_y = pd.concat([positive_target, negative_target], axis=0)
    pseudo_weight_array = np.concatenate([positive_weight, negative_weight]).astype("float64")
    return pseudo_x, pseudo_y, pseudo_weight_array


def _fit_eval_classifier(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    eval_family_mask: pd.Series,
    params: CatBoostHyperParams,
    early_stopping_rounds: int,
    verbose: int,
    sample_weight_train: Sequence[float] | pd.Series | None = None,
) -> dict[str, Any]:
    cat_columns = infer_categorical_columns(x_train)
    model = fit_with_validation(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        cat_columns=cat_columns,
        params=params,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
        sample_weight_train=sample_weight_train,
    )
    pred = predict_proba(model, x_eval)
    return {
        "global_auc": float(binary_auc(y_eval, pred)),
        "family_auc": _safe_auc_on_mask(y_eval, pred, eval_family_mask),
        "best_iteration": int(best_iteration_or_default(model, params.iterations)),
    }


def run_family_pseudo_label_experiment(
    *,
    train_csv_path: str | Path,
    family_key: str,
    reference_pred: pd.Series,
    reference_component_frame: pd.DataFrame | None,
    feature_blocks: Sequence[str] | None,
    params: CatBoostHyperParams,
    holdout_size: float,
    valid_size: float,
    pseudo_pool_fraction: float,
    repeats: int,
    random_state: int,
    early_stopping_rounds: int,
    verbose: int,
    upper_thresholds: Sequence[float],
    lower_thresholds: Sequence[float],
    pseudo_weights: Sequence[float],
    label_mode: str,
    scale_weight_by_confidence: bool,
    require_teacher_agreement: bool,
    max_teacher_std: float | None,
    min_selected_rows: int,
    metrics_path: str | Path,
    results_csv_path: str | Path,
) -> dict[str, Any]:
    """Run an offline family-level pseudo-labeling sweep."""
    if not (0.0 < float(holdout_size) < 0.5):
        raise ValueError("holdout_size must be in (0, 0.5)")
    if not (0.0 < float(valid_size) < 0.5):
        raise ValueError("valid_size must be in (0, 0.5)")
    if not (0.0 < float(pseudo_pool_fraction) < 0.9):
        raise ValueError("pseudo_pool_fraction must be in (0, 0.9)")
    if int(repeats) < 1:
        raise ValueError("repeats must be >= 1")

    train_df = load_csv(train_csv_path)
    normalized_blocks = normalize_feature_blocks(feature_blocks)
    stateless_blocks, stateful_blocks = partition_feature_blocks(normalized_blocks)
    x_base, y = prepare_train_features(train_df, drop_id=True, feature_blocks=stateless_blocks)
    family_series = build_segment5_family(train_df)
    family_mask = family_series.eq(str(family_key))
    family_rows = int(family_mask.sum())
    if family_rows < 2000:
        raise ValueError(f"family '{family_key}' has only {family_rows} rows; expected >= 2000")
    family_positive_rate = float(y.loc[family_mask].mean())

    reference_by_id = pd.Series(reference_pred, copy=True)
    aligned_reference_pred = train_df[ID_COLUMN].map(reference_by_id)
    if aligned_reference_pred.isna().any():
        missing_ids = train_df.loc[aligned_reference_pred.isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference_pred missing train ids: {missing_ids}")
    aligned_reference_pred = pd.Series(
        aligned_reference_pred.astype("float64").values,
        index=train_df.index,
        dtype="float64",
        name="reference_pred",
    )
    aligned_component_frame = _resolve_component_frame(train_df, reference_component_frame)
    if aligned_component_frame is not None:
        aligned_component_frame.index = train_df.index
    if (require_teacher_agreement or max_teacher_std is not None) and aligned_component_frame is None:
        raise ValueError(
            "Teacher agreement/std filtering requires per-model `pred_*` OOF components. "
            "Provide OOF inputs that merge into reference_component_frame."
        )

    config_rows: list[dict[str, Any]] = []
    config_grid = list(product(upper_thresholds, lower_thresholds, pseudo_weights))

    for repeat in range(int(repeats)):
        split_seed = int(random_state) + int(repeat)
        all_index = np.arange(len(train_df))
        stratify_labels = _build_split_stratify_labels(y, family_mask)
        fit_index, eval_index = train_test_split(
            all_index,
            test_size=float(holdout_size),
            random_state=split_seed,
            stratify=stratify_labels,
        )
        fit_index = pd.Index(sorted(fit_index))
        eval_index = pd.Index(sorted(eval_index))
        fit_family_index = fit_index[family_mask.loc[fit_index].to_numpy()]
        if len(fit_family_index) < 1000:
            raise ValueError(
                f"family '{family_key}' leaves only {len(fit_family_index)} fit rows in repeat {repeat + 1}"
            )

        pseudo_split_y = y.loc[fit_family_index]
        pseudo_stratify = pseudo_split_y if pseudo_split_y.nunique(dropna=False) >= 2 else None
        family_real_index, pseudo_pool_index = train_test_split(
            fit_family_index.to_numpy(),
            test_size=float(pseudo_pool_fraction),
            random_state=split_seed,
            stratify=pseudo_stratify,
        )
        family_real_index = pd.Index(sorted(family_real_index))
        pseudo_pool_index = pd.Index(sorted(pseudo_pool_index))
        train_real_index = fit_index.difference(pseudo_pool_index)

        train_real_y = y.loc[train_real_index]
        train_real_family_mask = family_mask.loc[train_real_index]
        inner_stratify = _build_split_stratify_labels(train_real_y, train_real_family_mask)
        inner_train_index, valid_index = train_test_split(
            train_real_index.to_numpy(),
            test_size=float(valid_size),
            random_state=split_seed,
            stratify=inner_stratify,
        )
        inner_train_index = pd.Index(sorted(inner_train_index))
        valid_index = pd.Index(sorted(valid_index))

        x_inner_train_base = x_base.loc[inner_train_index]
        x_valid_base = x_base.loc[valid_index]
        x_eval_base = x_base.loc[eval_index]
        x_pseudo_pool_base = x_base.loc[pseudo_pool_index]

        x_inner_train, transformed_views = _apply_stateful_views(
            x_inner_train_base,
            stateful_blocks,
            {
                "valid": x_valid_base,
                "eval": x_eval_base,
                "pseudo_pool": x_pseudo_pool_base,
            },
        )
        x_valid = transformed_views["valid"]
        x_eval = transformed_views["eval"]
        x_pseudo_pool = transformed_views["pseudo_pool"]

        y_inner_train = y.loc[inner_train_index]
        y_valid = y.loc[valid_index]
        y_eval = y.loc[eval_index]
        eval_family_mask = family_mask.loc[eval_index]

        pseudo_pool_reference = aligned_reference_pred.loc[pseudo_pool_index]
        pseudo_pool_truth = y.loc[pseudo_pool_index]
        pseudo_pool_component = None if aligned_component_frame is None else aligned_component_frame.loc[pseudo_pool_index]
        baseline_metrics = _fit_eval_classifier(
            x_train=x_inner_train,
            y_train=y_inner_train,
            x_valid=x_valid,
            y_valid=y_valid,
            x_eval=x_eval,
            y_eval=y_eval,
            eval_family_mask=eval_family_mask,
            params=params,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        for upper_threshold, lower_threshold, pseudo_weight in config_grid:
            selection_index, pseudo_target, selected_reference, selection_metrics = _select_pseudo_rows(
                y_true=pseudo_pool_truth,
                reference_pred=pseudo_pool_reference,
                component_frame=pseudo_pool_component,
                upper_threshold=float(upper_threshold),
                lower_threshold=float(lower_threshold),
                require_teacher_agreement=require_teacher_agreement,
                max_teacher_std=max_teacher_std,
                min_selected_rows=int(min_selected_rows),
            )

            row = {
                "repeat": int(repeat + 1),
                "split_seed": int(split_seed),
                "family_key": str(family_key),
                "upper_threshold": float(upper_threshold),
                "lower_threshold": float(lower_threshold),
                "pseudo_weight": float(pseudo_weight),
                "label_mode": str(label_mode),
                "scale_weight_by_confidence": bool(scale_weight_by_confidence),
                "holdout_rows": int(len(eval_index)),
                "holdout_family_rows": int(eval_family_mask.sum()),
                "pseudo_pool_rows": int(len(pseudo_pool_index)),
                "pseudo_pool_family_rows": int(len(pseudo_pool_index)),
                "train_real_rows": int(len(inner_train_index)),
                "valid_rows": int(len(valid_index)),
                **selection_metrics,
            }

            row["baseline_global_auc"] = baseline_metrics["global_auc"]
            row["baseline_family_auc"] = baseline_metrics["family_auc"]
            row["baseline_best_iteration"] = baseline_metrics["best_iteration"]

            if int(selection_metrics["selected_rows"]) < int(min_selected_rows):
                row["pseudo_global_auc"] = None
                row["pseudo_family_auc"] = None
                row["delta_global_auc"] = None
                row["delta_family_auc"] = None
                row["pseudo_best_iteration"] = None
                row["status"] = "skipped_low_selection"
                config_rows.append(row)
                continue

            x_selected_pseudo = x_pseudo_pool.loc[selection_index]
            pseudo_x, pseudo_y, pseudo_weight_array = _build_pseudo_training_rows(
                x_selected=x_selected_pseudo,
                pseudo_target=pseudo_target.loc[selection_index],
                selected_reference=selected_reference.loc[selection_index],
                pseudo_weight=float(pseudo_weight),
                label_mode=label_mode,
                scale_weight_by_confidence=scale_weight_by_confidence,
            )
            combined_x = pd.concat([x_inner_train, pseudo_x], axis=0, ignore_index=False)
            combined_y = pd.concat([y_inner_train, pseudo_y], axis=0, ignore_index=False)
            combined_weight = np.concatenate(
                [
                    np.ones(len(x_inner_train), dtype="float64"),
                    pseudo_weight_array,
                ]
            )
            fit_metrics = _fit_eval_classifier(
                x_train=combined_x,
                y_train=combined_y,
                x_valid=x_valid,
                y_valid=y_valid,
                x_eval=x_eval,
                y_eval=y_eval,
                eval_family_mask=eval_family_mask,
                params=params,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
                sample_weight_train=combined_weight,
            )
            row["pseudo_global_auc"] = fit_metrics["global_auc"]
            row["pseudo_family_auc"] = fit_metrics["family_auc"]
            row["delta_global_auc"] = float(fit_metrics["global_auc"] - baseline_metrics["global_auc"])
            row["delta_family_auc"] = (
                None
                if baseline_metrics["family_auc"] is None or fit_metrics["family_auc"] is None
                else float(fit_metrics["family_auc"] - baseline_metrics["family_auc"])
            )
            row["pseudo_best_iteration"] = fit_metrics["best_iteration"]
            row["status"] = "ok"
            config_rows.append(row)

    results_df = pd.DataFrame(config_rows)
    results_out = ensure_parent_dir(results_csv_path)
    results_df.to_csv(results_out, index=False)

    valid_rows = results_df.loc[results_df["status"].eq("ok")].copy()
    summary_rows: list[dict[str, Any]] = []
    if not valid_rows.empty:
        group_columns = ["upper_threshold", "lower_threshold", "pseudo_weight"]
        for (upper_threshold, lower_threshold, pseudo_weight), part in valid_rows.groupby(group_columns, dropna=False):
            summary_rows.append(
                {
                    "upper_threshold": float(upper_threshold),
                    "lower_threshold": float(lower_threshold),
                    "pseudo_weight": float(pseudo_weight),
                    "repeats": int(len(part)),
                    "mean_delta_global_auc": float(part["delta_global_auc"].mean()),
                    "std_delta_global_auc": float(part["delta_global_auc"].std(ddof=0)),
                    "mean_delta_family_auc": (
                        None
                        if part["delta_family_auc"].dropna().empty
                        else float(part["delta_family_auc"].dropna().mean())
                    ),
                    "mean_selected_rows": float(part["selected_rows"].mean()),
                    "mean_selected_accuracy": (
                        None
                        if part["selected_accuracy"].dropna().empty
                        else float(part["selected_accuracy"].dropna().mean())
                    ),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["mean_delta_global_auc", "mean_delta_family_auc", "mean_selected_accuracy"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    payload: dict[str, Any] = {
        "train_csv_path": str(train_csv_path),
        "family_key": str(family_key),
        "family_rows": int(family_rows),
        "family_positive_rate": float(family_positive_rate),
        "feature_blocks": list(normalized_blocks),
        "holdout_size": float(holdout_size),
        "valid_size": float(valid_size),
        "pseudo_pool_fraction": float(pseudo_pool_fraction),
        "repeats": int(repeats),
        "random_state": int(random_state),
        "require_teacher_agreement": bool(require_teacher_agreement),
        "max_teacher_std": None if max_teacher_std is None else float(max_teacher_std),
        "min_selected_rows": int(min_selected_rows),
        "upper_thresholds": [float(value) for value in upper_thresholds],
        "lower_thresholds": [float(value) for value in lower_thresholds],
        "pseudo_weights": [float(value) for value in pseudo_weights],
        "label_mode": str(label_mode),
        "scale_weight_by_confidence": bool(scale_weight_by_confidence),
        "results_csv_path": str(results_out),
        "config_count": int(len(config_grid)),
        "evaluated_runs": int(len(valid_rows)),
        "best_config": None if summary_df.empty else summary_df.iloc[0].to_dict(),
        "config_summary": summary_df.to_dict(orient="records"),
    }
    write_json(metrics_path, payload)
    return payload
