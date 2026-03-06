"""Utilities for comparing model OOF predictions and building blends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from .config import ID_COLUMN
from .evaluation import binary_auc

OOF_TARGET_COLUMN = "target"


@dataclass(frozen=True)
class OOFInput:
    """Input descriptor for one model OOF artifact."""

    name: str
    path: str
    prediction_column: str | None = None


def parse_oof_input(raw: str) -> OOFInput:
    """Parse '<name>=<path>[#<prediction_column>]' into OOFInput."""
    text = raw.strip()
    if "=" not in text:
        raise ValueError(
            f"Invalid --oof value '{raw}'. Expected format: <name>=<path>[#<prediction_column>]"
        )

    name, payload = text.split("=", 1)
    name = name.strip()
    payload = payload.strip()
    if not name:
        raise ValueError(f"Invalid --oof value '{raw}': empty model name")
    if not payload:
        raise ValueError(f"Invalid --oof value '{raw}': empty path")

    path_value, has_column, prediction_column = payload.partition("#")
    path_value = path_value.strip()
    prediction_column = prediction_column.strip() if has_column else None
    if not path_value:
        raise ValueError(f"Invalid --oof value '{raw}': empty path")
    if prediction_column == "":
        prediction_column = None

    return OOFInput(name=name, path=path_value, prediction_column=prediction_column)


def detect_prediction_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    """Resolve prediction column from common OOF naming conventions."""
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Prediction column '{preferred}' not found in OOF file columns")
        return preferred

    for candidate in ("oof_pred", "oof_ensemble", "prediction", "pred"):
        if candidate in df.columns:
            return candidate

    oof_like = [column for column in df.columns if str(column).startswith("oof_")]
    if len(oof_like) == 1:
        return oof_like[0]
    if len(oof_like) > 1:
        raise ValueError(
            "Could not auto-detect prediction column (multiple oof_* columns found): "
            f"{oof_like}. Use #<prediction_column> in --oof."
        )
    raise ValueError(
        "Could not auto-detect prediction column. Expected one of "
        "'oof_pred', 'oof_ensemble', 'prediction', 'pred', or exactly one 'oof_*' column."
    )


def load_oof_frame(
    item: OOFInput,
    *,
    id_column: str = ID_COLUMN,
    target_column: str = OOF_TARGET_COLUMN,
) -> tuple[pd.DataFrame, str]:
    """Load one OOF frame and normalize prediction column name to 'pred_<name>'."""
    df = pd.read_csv(item.path)
    required = [id_column, target_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"OOF file '{item.path}' is missing required columns: {missing}")

    pred_column = detect_prediction_column(df, preferred=item.prediction_column)
    model_column = f"pred_{item.name}"
    out = df[[id_column, target_column, pred_column]].copy()
    out = out.rename(columns={pred_column: model_column})
    return out, model_column


def load_merged_oof(
    items: Iterable[OOFInput],
    *,
    id_column: str = ID_COLUMN,
    target_column: str = OOF_TARGET_COLUMN,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge multiple OOF frames on id+target and return merged frame + model columns."""
    parsed = list(items)
    if not parsed:
        raise ValueError("At least one OOF input is required.")

    merged = None
    model_columns: list[str] = []
    for item in parsed:
        frame, model_column = load_oof_frame(item, id_column=id_column, target_column=target_column)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(
                frame,
                on=[id_column, target_column],
                how="inner",
                validate="one_to_one",
            )
        model_columns.append(model_column)

    if merged is None:
        raise RuntimeError("Unexpected empty merged OOF dataframe.")
    if merged.empty:
        raise ValueError("Merged OOF dataframe is empty after joins. Check id consistency.")
    return merged, model_columns


def compute_model_auc_table(
    merged: pd.DataFrame,
    model_columns: list[str],
    *,
    target_column: str = OOF_TARGET_COLUMN,
) -> list[dict[str, float | str | int]]:
    """Compute per-model AUCs from merged OOF data."""
    y_true = merged[target_column].astype(int).values
    rows = []
    for column in model_columns:
        auc = binary_auc(y_true, merged[column].values)
        rows.append({"model_column": column, "oof_auc": float(auc)})

    rows = sorted(rows, key=lambda item: float(item["oof_auc"]), reverse=True)
    best_auc = float(rows[0]["oof_auc"])
    for idx, row in enumerate(rows, start=1):
        row["rank"] = int(idx)
        row["delta_vs_best"] = float(row["oof_auc"] - best_auc)
    return rows


def compute_oof_correlation(
    merged: pd.DataFrame,
    model_columns: list[str],
) -> pd.DataFrame:
    """Compute Pearson correlation matrix for OOF predictions."""
    return merged[model_columns].corr(method="pearson")


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Project non-negative weights to simplex."""
    clipped = np.maximum(weights.astype("float64"), 0.0)
    weight_sum = float(np.sum(clipped))
    if weight_sum <= 0.0:
        return np.full(shape=clipped.shape, fill_value=1.0 / float(len(clipped)))
    return clipped / weight_sum


def blend_predictions(
    merged: pd.DataFrame,
    model_columns: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    """Build blended prediction from model columns and simplex weights."""
    if len(model_columns) != len(weights):
        raise ValueError("weights length must match model column count")
    matrix = merged[model_columns].to_numpy(dtype="float64")
    return matrix @ normalize_weights(weights)


def blend_auc(
    merged: pd.DataFrame,
    model_columns: list[str],
    weights: np.ndarray,
    *,
    target_column: str = OOF_TARGET_COLUMN,
) -> float:
    """Evaluate blend AUC for given weights."""
    y_true = merged[target_column].astype(int).values
    pred = blend_predictions(merged, model_columns, weights)
    return binary_auc(y_true, pred)


def _integer_compositions(total: int, parts: int) -> Iterable[list[int]]:
    """Yield all integer compositions for simplex grid search."""
    if parts == 1:
        yield [total]
        return
    for value in range(total + 1):
        for tail in _integer_compositions(total - value, parts - 1):
            yield [value, *tail]


def grid_search_blend_weights(
    merged: pd.DataFrame,
    model_columns: list[str],
    *,
    grid_step: float = 0.05,
    top_k: int = 5,
    target_column: str = OOF_TARGET_COLUMN,
) -> list[dict[str, object]]:
    """Brute-force simplex search over blend weights (recommended for <=4 models)."""
    if grid_step <= 0 or grid_step > 1:
        raise ValueError("grid_step must be in (0, 1].")
    if len(model_columns) < 2:
        raise ValueError("Grid search requires at least 2 models.")
    if len(model_columns) > 4:
        raise ValueError("Grid search is limited to at most 4 models.")

    inv_step = 1.0 / float(grid_step)
    if abs(round(inv_step) - inv_step) > 1e-9:
        raise ValueError("grid_step must be an exact divisor of 1.0 (e.g., 0.1, 0.05, 0.02).")
    total = int(round(inv_step))

    y_true = merged[target_column].astype(int).values
    matrix = merged[model_columns].to_numpy(dtype="float64")

    scored: list[dict[str, object]] = []
    for composition in _integer_compositions(total, len(model_columns)):
        weights = np.array(composition, dtype="float64") / float(total)
        pred = matrix @ weights
        auc = binary_auc(y_true, pred)
        scored.append(
            {
                "auc": float(auc),
                "weights": {
                    model_columns[idx]: float(weights[idx]) for idx in range(len(model_columns))
                },
            }
        )

    scored = sorted(scored, key=lambda item: float(item["auc"]), reverse=True)
    return scored[: int(max(top_k, 1))]


def optimize_blend_coordinate_descent(
    merged: pd.DataFrame,
    model_columns: list[str],
    *,
    step: float = 0.02,
    max_rounds: int = 25,
    target_column: str = OOF_TARGET_COLUMN,
) -> dict[str, object]:
    """Simplex-constrained coordinate descent for blend weights."""
    if len(model_columns) < 2:
        raise ValueError("Coordinate descent requires at least 2 models.")
    if step <= 0 or step >= 1:
        raise ValueError("step must be in (0, 1).")
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1.")

    matrix = merged[model_columns].to_numpy(dtype="float64")
    y_true = merged[target_column].astype(int).values

    n_models = len(model_columns)
    weights = np.full(shape=n_models, fill_value=1.0 / float(n_models), dtype="float64")
    best_auc = binary_auc(y_true, matrix @ weights)

    rounds_used = 0
    for round_idx in range(max_rounds):
        improved = False
        for focus in range(n_models):
            for direction in (-1.0, 1.0):
                candidate = weights.copy()
                old_focus = float(candidate[focus])
                new_focus = old_focus + float(direction) * float(step)
                if new_focus < 0.0 or new_focus > 1.0:
                    continue

                if n_models == 1:
                    candidate[focus] = 1.0
                else:
                    rest_old = 1.0 - old_focus
                    rest_new = 1.0 - new_focus
                    candidate[focus] = new_focus
                    if rest_old <= 1e-12:
                        fill = rest_new / float(n_models - 1)
                        for idx in range(n_models):
                            if idx != focus:
                                candidate[idx] = fill
                    else:
                        scale = rest_new / rest_old
                        for idx in range(n_models):
                            if idx != focus:
                                candidate[idx] = candidate[idx] * scale
                candidate = normalize_weights(candidate)
                auc = binary_auc(y_true, matrix @ candidate)
                if float(auc) > float(best_auc) + 1e-10:
                    weights = candidate
                    best_auc = float(auc)
                    improved = True
        rounds_used = round_idx + 1
        if not improved:
            break

    return {
        "auc": float(best_auc),
        "rounds_used": int(rounds_used),
        "weights": {model_columns[idx]: float(weights[idx]) for idx in range(n_models)},
    }


def stack_oof_cross_fitted(
    merged: pd.DataFrame,
    model_columns: list[str],
    *,
    stacker: str = "logistic",
    folds: int = 5,
    random_state: int = 42,
    logistic_c: float = 1.0,
    logistic_max_iter: int = 2000,
    ridge_alpha: float = 1.0,
    target_column: str = OOF_TARGET_COLUMN,
) -> dict[str, object]:
    """Build cross-fitted stacker predictions from OOF model columns."""
    if len(model_columns) < 2:
        raise ValueError("Stacking requires at least 2 base models.")
    if folds < 2:
        raise ValueError("folds must be >= 2")

    normalized_stacker = str(stacker).strip().lower()
    if normalized_stacker not in {"logistic", "ridge"}:
        raise ValueError("stacker must be one of: logistic, ridge")

    x = merged[model_columns].to_numpy(dtype="float64")
    y = merged[target_column].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(random_state))

    oof_stack = np.zeros(shape=len(merged), dtype="float64")
    fold_rows: list[dict[str, object]] = []

    def _fit_stacker(x_fit: np.ndarray, y_fit: np.ndarray):
        if normalized_stacker == "logistic":
            model = LogisticRegression(
                C=float(logistic_c),
                solver="lbfgs",
                max_iter=int(logistic_max_iter),
            )
        else:
            model = Ridge(
                alpha=float(ridge_alpha),
                random_state=int(random_state),
            )
        model.fit(x_fit, y_fit)
        return model

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        model = _fit_stacker(x[train_idx], y[train_idx])

        if normalized_stacker == "logistic":
            pred_valid = model.predict_proba(x[valid_idx])[:, 1]
            coefficients = model.coef_[0].tolist()
            intercept = float(model.intercept_[0])
        else:
            pred_valid = np.clip(model.predict(x[valid_idx]), a_min=0.0, a_max=1.0)
            coefficients = model.coef_.tolist()
            intercept = float(model.intercept_)

        oof_stack[valid_idx] = pred_valid
        fold_auc = binary_auc(y[valid_idx], pred_valid)
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "auc": float(fold_auc),
                "stacker": normalized_stacker,
                "intercept": intercept,
                "coefficients": {
                    model_columns[idx]: float(coefficients[idx]) for idx in range(len(model_columns))
                },
            }
        )

    oof_auc = binary_auc(y, oof_stack)
    fold_aucs = [float(row["auc"]) for row in fold_rows]

    full_model = _fit_stacker(x, y)
    if normalized_stacker == "logistic":
        full_coef = full_model.coef_[0].tolist()
        full_intercept = float(full_model.intercept_[0])
    else:
        full_coef = full_model.coef_.tolist()
        full_intercept = float(full_model.intercept_)

    return {
        "stacker": normalized_stacker,
        "oof_pred": oof_stack,
        "oof_auc": float(oof_auc),
        "cv_mean_auc": float(np.mean(fold_aucs)),
        "cv_std_auc": float(np.std(fold_aucs)),
        "cv_fold_metrics": fold_rows,
        "full_intercept": full_intercept,
        "full_coefficients": {
            model_columns[idx]: float(full_coef[idx]) for idx in range(len(model_columns))
        },
        "params": {
            "folds": int(folds),
            "random_state": int(random_state),
            "logistic_c": float(logistic_c),
            "logistic_max_iter": int(logistic_max_iter),
            "ridge_alpha": float(ridge_alpha),
        },
    }


def load_baseline_auc(metrics_path: str | Path) -> tuple[float, str]:
    """Load baseline AUC from known metric keys."""
    import json

    with Path(metrics_path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    for key in ("ensemble_oof_auc", "oof_auc", "holdout_auc"):
        if key in payload:
            return float(payload[key]), key
    raise ValueError(
        f"Could not find baseline AUC in {metrics_path}. "
        "Expected one of: ensemble_oof_auc, oof_auc, holdout_auc."
    )
