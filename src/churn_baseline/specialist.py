"""Specialist-model experiments for hard churn cohorts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .artifacts import ensure_parent_dir, write_json
from .config import CatBoostHyperParams, ID_COLUMN, TARGET_COLUMN
from .data import infer_categorical_columns, load_csv
from .evaluation import binary_auc
from .feature_engineering import normalize_feature_blocks, partition_feature_blocks
from .modeling import best_iteration_or_default, fit_full_train, fit_with_validation, load_model, predict_proba, save_model
from .pipeline import _prepare_test_matrix, _prepare_train_matrix, _transform_pair_with_stateful_blocks, _transform_single_with_stateful_blocks


EARLY_MANUAL_INTERNET = "early_manual_internet"
EARLY_ALL_INTERNET = "early_all_internet"
FIBER_PAPERLESS_EARLY = "fiber_paperless_early"
LATE_MTM_FIBER = "late_mtm_fiber"
LATE_MTM_FIBER_PAPERLESS = "late_mtm_fiber_paperless"
LONG_FIBER_ANY = "long_fiber_any"
MANUAL_NO_INTERNET = "manual_no_internet"
ONE_YEAR_FIBER_ANY = "one_year_fiber_any"
ONE_YEAR_DSL_ANY = "one_year_dsl_any"
ONE_YEAR_DSL_PAPERLESS_49PLUS = "one_year_dsl_paperless_49plus"
TWO_YEAR_FIBER_ANY = "two_year_fiber_any"
TWO_YEAR_DSL_PAPERLESS_49PLUS = "two_year_dsl_paperless_49plus"
ONE_YEAR_FIBER_PAPERLESS_49PLUS = "one_year_fiber_paperless_49plus"
ONE_YEAR_FIBER_PAPERLESS_25_48 = "one_year_fiber_paperless_25_48"
TWO_YEAR_FIBER_PAPERLESS_49PLUS = "two_year_fiber_paperless_49plus"
TWO_YEAR_FIBER_NOPAPERLESS_49PLUS = "two_year_fiber_nopaperless_49plus"
MTM_DSL_PAPERLESS_25_48_MANUAL = "mtm_dsl_paperless_25_48_manual"
MTM_DSL_PAPERLESS_25_48_ANY = "mtm_dsl_paperless_25_48_any"
MTM_NOINTERNET_MAILED_0_24 = "mtm_nointernet_mailed_0_24"
MTM_NOINTERNET_NO_0_6 = "mtm_nointernet_no_0_6"
EC_MTM_FIBER_PAPERLESS_0_6 = "ec_mtm_fiber_paperless_0_6"
EC_MTM_FIBER_PAPERLESS_25_48 = "ec_mtm_fiber_paperless_25_48"
EC_MTM_FIBER_PAPERLESS_ANY = "ec_mtm_fiber_paperless_any"
EC_MTM_FIBER_ANY = "ec_mtm_fiber_any"
EC_MTM_DSL_PAPERLESS_0_6 = "ec_mtm_dsl_paperless_0_6"
EC_MTM_DSL_PAPERLESS_ANY = "ec_mtm_dsl_paperless_any"
CLASSIFIER = "classifier"
RESIDUAL = "residual"

SPECIALIST_PRESETS: dict[str, str] = {
    EARLY_MANUAL_INTERNET: (
        "Month-to-month, internet activo, tenure <= 24, payment manual "
        "(Electronic check o Mailed check)."
    ),
    EARLY_ALL_INTERNET: "Month-to-month, internet activo, tenure <= 24.",
    FIBER_PAPERLESS_EARLY: (
        "Month-to-month, Fiber optic, PaperlessBilling=Yes, tenure <= 24."
    ),
    LATE_MTM_FIBER: "Month-to-month, Fiber optic, tenure > 24.",
    LATE_MTM_FIBER_PAPERLESS: (
        "Month-to-month, Fiber optic, PaperlessBilling=Yes, tenure > 24."
    ),
    LONG_FIBER_ANY: "One year o Two year, Fiber optic, tenure > 24.",
    MANUAL_NO_INTERNET: "InternetService=No con payment manual (Electronic o Mailed check).",
    ONE_YEAR_FIBER_ANY: "One year, Fiber optic, cualquier tenure.",
    ONE_YEAR_DSL_ANY: "One year, DSL, cualquier tenure.",
    ONE_YEAR_DSL_PAPERLESS_49PLUS: "One year, DSL, PaperlessBilling=Yes, tenure >= 49.",
    TWO_YEAR_FIBER_ANY: "Two year, Fiber optic, cualquier tenure.",
    TWO_YEAR_DSL_PAPERLESS_49PLUS: "Two year, DSL, PaperlessBilling=Yes, tenure >= 49.",
    ONE_YEAR_FIBER_PAPERLESS_49PLUS: "One year, Fiber optic, PaperlessBilling=Yes, tenure >= 49.",
    ONE_YEAR_FIBER_PAPERLESS_25_48: "One year, Fiber optic, PaperlessBilling=Yes, tenure 25-48.",
    TWO_YEAR_FIBER_PAPERLESS_49PLUS: "Two year, Fiber optic, PaperlessBilling=Yes, tenure >= 49.",
    TWO_YEAR_FIBER_NOPAPERLESS_49PLUS: "Two year, Fiber optic, PaperlessBilling=No, tenure >= 49.",
    MTM_DSL_PAPERLESS_25_48_MANUAL: (
        "Month-to-month, DSL, PaperlessBilling=Yes, tenure 25-48, "
        "payment manual (Electronic check o Mailed check)."
    ),
    MTM_DSL_PAPERLESS_25_48_ANY: (
        "Month-to-month, DSL, PaperlessBilling=Yes, tenure 25-48, "
        "cualquier payment method."
    ),
    MTM_NOINTERNET_MAILED_0_24: (
        "Month-to-month, InternetService=No, Mailed check, tenure <= 24."
    ),
    MTM_NOINTERNET_NO_0_6: (
        "Month-to-month, InternetService=No, PaperlessBilling=No, tenure <= 6."
    ),
    EC_MTM_FIBER_PAPERLESS_0_6: (
        "Electronic check, Month-to-month, Fiber optic, PaperlessBilling=Yes, tenure <= 6."
    ),
    EC_MTM_FIBER_PAPERLESS_25_48: (
        "Electronic check, Month-to-month, Fiber optic, PaperlessBilling=Yes, tenure 25-48."
    ),
    EC_MTM_FIBER_PAPERLESS_ANY: (
        "Electronic check, Month-to-month, Fiber optic, PaperlessBilling=Yes, cualquier tenure."
    ),
    EC_MTM_FIBER_ANY: (
        "Electronic check, Month-to-month, Fiber optic, cualquier PaperlessBilling, cualquier tenure."
    ),
    EC_MTM_DSL_PAPERLESS_0_6: (
        "Electronic check, Month-to-month, DSL, PaperlessBilling=Yes, tenure <= 6."
    ),
    EC_MTM_DSL_PAPERLESS_ANY: (
        "Electronic check, Month-to-month, DSL, PaperlessBilling=Yes, cualquier tenure."
    ),
}

_MIN_SPECIALIST_TRAIN_ROWS = 2000
_MIN_SPECIALIST_VALID_ROWS = 500
PLATT = "platt"
ISOTONIC = "isotonic"
CALIBRATION_METHODS: tuple[str, ...] = (PLATT, ISOTONIC)
SPECIALIST_APPROACHES: tuple[str, ...] = (CLASSIFIER, RESIDUAL)
_TEACHER_COMPONENT_PREFIX = "teacher_component_"


def list_specialist_presets() -> dict[str, str]:
    """Return supported specialist presets and descriptions."""
    return dict(SPECIALIST_PRESETS)


def list_calibration_methods() -> tuple[str, ...]:
    """Return supported local calibration methods."""
    return CALIBRATION_METHODS


def list_specialist_approaches() -> tuple[str, ...]:
    """Return supported local specialist experiment approaches."""
    return SPECIALIST_APPROACHES


def normalize_binary_target(target: pd.Series) -> pd.Series:
    """Normalize binary target to 0/1 integers."""
    if pd.api.types.is_numeric_dtype(target):
        unique_values = set(pd.Series(target).dropna().astype(int).unique().tolist())
        if unique_values.issubset({0, 1}):
            return pd.Series(target, copy=True).astype(int)

    mapped = target.astype(str).map({"Yes": 1, "No": 0})
    if mapped.isna().any():
        sample = target.loc[mapped.isna()].astype(str).unique().tolist()[:5]
        raise ValueError(f"Unsupported target labels for binary normalization: {sample}")
    return mapped.astype(int)


def build_specialist_mask(frame: pd.DataFrame, preset: str) -> pd.Series:
    """Build boolean mask for a supported specialist preset."""
    required = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "tenure"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Specialist preset '{preset}' requires columns: {missing}")

    contract = frame["Contract"].astype(str)
    internet = frame["InternetService"].astype(str)
    payment = frame["PaymentMethod"].astype(str)
    paperless = frame["PaperlessBilling"].astype(str)
    tenure = pd.to_numeric(frame["tenure"], errors="coerce").fillna(0.0)

    if preset == EARLY_MANUAL_INTERNET:
        return (
            contract.eq("Month-to-month")
            & internet.ne("No")
            & tenure.le(24)
            & payment.isin(["Electronic check", "Mailed check"])
        )
    if preset == EARLY_ALL_INTERNET:
        return contract.eq("Month-to-month") & internet.ne("No") & tenure.le(24)
    if preset == FIBER_PAPERLESS_EARLY:
        return (
            contract.eq("Month-to-month")
            & internet.eq("Fiber optic")
            & paperless.eq("Yes")
            & tenure.le(24)
        )
    if preset == LATE_MTM_FIBER:
        return contract.eq("Month-to-month") & internet.eq("Fiber optic") & tenure.gt(24)
    if preset == LATE_MTM_FIBER_PAPERLESS:
        return (
            contract.eq("Month-to-month")
            & internet.eq("Fiber optic")
            & paperless.eq("Yes")
            & tenure.gt(24)
        )
    if preset == LONG_FIBER_ANY:
        return contract.isin(["One year", "Two year"]) & internet.eq("Fiber optic") & tenure.gt(24)
    if preset == MANUAL_NO_INTERNET:
        return internet.eq("No") & payment.isin(["Electronic check", "Mailed check"])
    if preset == ONE_YEAR_FIBER_ANY:
        return contract.eq("One year") & internet.eq("Fiber optic")
    if preset == ONE_YEAR_DSL_ANY:
        return contract.eq("One year") & internet.eq("DSL")
    if preset == ONE_YEAR_DSL_PAPERLESS_49PLUS:
        return contract.eq("One year") & internet.eq("DSL") & paperless.eq("Yes") & tenure.ge(49)
    if preset == TWO_YEAR_FIBER_ANY:
        return contract.eq("Two year") & internet.eq("Fiber optic")
    if preset == TWO_YEAR_DSL_PAPERLESS_49PLUS:
        return contract.eq("Two year") & internet.eq("DSL") & paperless.eq("Yes") & tenure.ge(49)
    if preset == ONE_YEAR_FIBER_PAPERLESS_49PLUS:
        return contract.eq("One year") & internet.eq("Fiber optic") & paperless.eq("Yes") & tenure.ge(49)
    if preset == ONE_YEAR_FIBER_PAPERLESS_25_48:
        return contract.eq("One year") & internet.eq("Fiber optic") & paperless.eq("Yes") & tenure.between(25, 48)
    if preset == TWO_YEAR_FIBER_PAPERLESS_49PLUS:
        return contract.eq("Two year") & internet.eq("Fiber optic") & paperless.eq("Yes") & tenure.ge(49)
    if preset == TWO_YEAR_FIBER_NOPAPERLESS_49PLUS:
        return contract.eq("Two year") & internet.eq("Fiber optic") & paperless.eq("No") & tenure.ge(49)
    if preset == MTM_DSL_PAPERLESS_25_48_MANUAL:
        return (
            contract.eq("Month-to-month")
            & internet.eq("DSL")
            & paperless.eq("Yes")
            & tenure.between(25, 48)
            & payment.isin(["Electronic check", "Mailed check"])
        )
    if preset == MTM_DSL_PAPERLESS_25_48_ANY:
        return contract.eq("Month-to-month") & internet.eq("DSL") & paperless.eq("Yes") & tenure.between(25, 48)
    if preset == MTM_NOINTERNET_MAILED_0_24:
        return (
            contract.eq("Month-to-month")
            & internet.eq("No")
            & payment.eq("Mailed check")
            & tenure.le(24)
        )
    if preset == MTM_NOINTERNET_NO_0_6:
        return (
            contract.eq("Month-to-month")
            & internet.eq("No")
            & paperless.eq("No")
            & tenure.le(6)
        )
    if preset == EC_MTM_FIBER_PAPERLESS_0_6:
        return (
            contract.eq("Month-to-month")
            & internet.eq("Fiber optic")
            & payment.eq("Electronic check")
            & paperless.eq("Yes")
            & tenure.le(6)
        )
    if preset == EC_MTM_FIBER_PAPERLESS_25_48:
        return (
            contract.eq("Month-to-month")
            & internet.eq("Fiber optic")
            & payment.eq("Electronic check")
            & paperless.eq("Yes")
            & tenure.between(25, 48)
        )
    if preset == EC_MTM_FIBER_PAPERLESS_ANY:
        return (
            contract.eq("Month-to-month")
            & internet.eq("Fiber optic")
            & payment.eq("Electronic check")
            & paperless.eq("Yes")
        )
    if preset == EC_MTM_FIBER_ANY:
        return (
            contract.eq("Month-to-month")
            & internet.eq("Fiber optic")
            & payment.eq("Electronic check")
        )
    if preset == EC_MTM_DSL_PAPERLESS_0_6:
        return (
            contract.eq("Month-to-month")
            & internet.eq("DSL")
            & payment.eq("Electronic check")
            & paperless.eq("Yes")
            & tenure.le(6)
        )
    if preset == EC_MTM_DSL_PAPERLESS_ANY:
        return (
            contract.eq("Month-to-month")
            & internet.eq("DSL")
            & payment.eq("Electronic check")
            & paperless.eq("Yes")
        )
    raise ValueError(f"Unsupported specialist preset '{preset}'. Available: {sorted(SPECIALIST_PRESETS)}")


def build_reference_prediction(
    merged_oof: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Build weighted reference prediction from merged OOF matrix."""
    reference = pd.Series(0.0, index=merged_oof.index, dtype="float64")
    missing = []
    for name, weight in weights.items():
        column = f"pred_{name}"
        if column not in merged_oof.columns:
            missing.append(column)
            continue
        reference = reference + float(weight) * merged_oof[column].astype("float64")
    if missing:
        raise ValueError(f"Missing weighted OOF columns: {missing}")
    return pd.Series(reference.values, index=merged_oof[ID_COLUMN].values, dtype="float64", name="reference_pred")


def _save_pickle(path: str | Path, payload: Any) -> Path:
    out_path = ensure_parent_dir(path)
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return out_path


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def _prob_to_logit(probabilities: pd.Series | np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype="float64"), epsilon, 1.0 - epsilon)
    return np.log(probs / (1.0 - probs))


def _clip_probability(pred: pd.Series | np.ndarray, epsilon: float = 1e-6) -> pd.Series:
    clipped = np.clip(np.asarray(pred, dtype="float64"), epsilon, 1.0 - epsilon)
    return pd.Series(clipped, index=getattr(pred, "index", None), dtype="float64")


def _normalize_reference_component_frame(
    ids: pd.Series,
    reference_component_frame: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if reference_component_frame is None:
        return None
    if ID_COLUMN not in reference_component_frame.columns:
        raise ValueError(f"reference_component_frame must contain column '{ID_COLUMN}'")

    component_columns = [
        column
        for column in reference_component_frame.columns
        if column != ID_COLUMN and str(column).startswith("pred_")
    ]
    if not component_columns:
        raise ValueError("reference_component_frame must contain at least one 'pred_*' column")

    component_frame = ids.to_frame(name=ID_COLUMN).merge(
        reference_component_frame[[ID_COLUMN, *component_columns]].copy(),
        how="left",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if component_frame[component_columns].isna().any().any():
        missing_ids = component_frame.loc[
            component_frame[component_columns].isna().any(axis=1),
            ID_COLUMN,
        ].head(5).tolist()
        raise ValueError(f"reference_component_frame missing ids: {missing_ids}")
    return component_frame


def _append_teacher_disagreement_features(
    x: pd.DataFrame,
    reference_pred: pd.Series,
    reference_component_frame: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    if reference_component_frame is None:
        return x, []

    component_columns = [column for column in reference_component_frame.columns if column != ID_COLUMN]
    out = x.copy()
    component_matrix = reference_component_frame[component_columns].to_numpy(dtype="float64")
    disagreement_columns: list[str] = []

    for column in component_columns:
        feature_name = f"{_TEACHER_COMPONENT_PREFIX}{str(column).removeprefix('pred_')}"
        out[feature_name] = reference_component_frame[column].astype("float64").values
        disagreement_columns.append(feature_name)

    component_mean = component_matrix.mean(axis=1)
    component_std = component_matrix.std(axis=1)
    component_min = component_matrix.min(axis=1)
    component_max = component_matrix.max(axis=1)
    sorted_components = np.sort(component_matrix, axis=1)
    top_gap = sorted_components[:, -1] - sorted_components[:, -2] if component_matrix.shape[1] >= 2 else np.zeros(len(out), dtype="float64")

    derived_features = {
        "teacher_component_mean": component_mean,
        "teacher_component_std": component_std,
        "teacher_component_range": component_max - component_min,
        "teacher_component_top_gap": top_gap,
        "teacher_reference_minus_component_mean": reference_pred.astype("float64").values - component_mean,
    }

    component_feature_lookup = {
        str(column).removeprefix("pred_"): f"{_TEACHER_COMPONENT_PREFIX}{str(column).removeprefix('pred_')}"
        for column in component_columns
    }
    pair_candidates = (("cb", "xgb"), ("cb", "rv"), ("xgb", "rv"), ("cb", "r"), ("xgb", "r"))
    for left_name, right_name in pair_candidates:
        left_feature = component_feature_lookup.get(left_name)
        right_feature = component_feature_lookup.get(right_name)
        if left_feature is None or right_feature is None:
            continue
        delta = out[left_feature].astype("float64") - out[right_feature].astype("float64")
        feature_base = f"teacher_{left_name}_minus_{right_name}"
        derived_features[feature_base] = delta.values
        derived_features[f"{feature_base}_abs"] = delta.abs().values

    for feature_name, values in derived_features.items():
        out[feature_name] = np.asarray(values, dtype="float64")
        disagreement_columns.append(feature_name)

    return out, disagreement_columns


def _append_reference_features(
    x: pd.DataFrame,
    *,
    reference_pred: pd.Series,
    reference_component_frame: pd.DataFrame | None,
    include_logit: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Append reference and optional teacher-disagreement features to a matrix."""
    out = x.copy()
    out["reference_pred_feature"] = reference_pred.astype("float64").values
    if include_logit:
        out["reference_logit_feature"] = _prob_to_logit(reference_pred)
    return _append_teacher_disagreement_features(out, reference_pred, reference_component_frame)


def _prepare_specialist_test_matrix(
    *,
    test_df: pd.DataFrame,
    feature_blocks: Sequence[str] | None,
    train_csv_path: str | Path | None,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Prepare specialist inference features, including fit-aware blocks when requested."""
    normalized_blocks = normalize_feature_blocks(feature_blocks)
    stateless_blocks, stateful_blocks = partition_feature_blocks(normalized_blocks)
    if stateful_blocks and train_csv_path is None:
        raise ValueError(
            "train_csv_path is required for specialist inference when using fit-aware "
            f"feature blocks: {list(stateful_blocks)}"
        )
    x_test = _prepare_test_matrix(
        test_df,
        stateless_blocks=stateless_blocks,
        stateful_blocks=stateful_blocks,
        train_csv_path=train_csv_path,
    )
    return normalized_blocks, x_test


def _fit_local_calibrator(method: str, pred: pd.Series, y_true: pd.Series) -> dict[str, Any]:
    method_value = str(method).strip().lower()
    if method_value == PLATT:
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(_prob_to_logit(pred).reshape(-1, 1), y_true.astype(int))
        return {"method": method_value, "model": model}
    if method_value == ISOTONIC:
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(np.asarray(pred, dtype="float64"), y_true.astype(int))
        return {"method": method_value, "model": model}
    raise ValueError(f"Unsupported calibration method '{method}'. Available: {CALIBRATION_METHODS}")


def _predict_local_calibrator(bundle: dict[str, Any], pred: pd.Series | np.ndarray) -> pd.Series:
    method_value = str(bundle["method"]).strip().lower()
    model = bundle["model"]
    pred_array = np.asarray(pred, dtype="float64")
    if method_value == PLATT:
        calibrated = model.predict_proba(_prob_to_logit(pred_array).reshape(-1, 1))[:, 1]
    elif method_value == ISOTONIC:
        calibrated = model.predict(pred_array)
    else:
        raise ValueError(f"Unsupported calibration method '{method_value}'.")
    return pd.Series(calibrated, index=getattr(pred, "index", None), dtype="float64")


def run_specialist_override_cv(
    *,
    train_csv_path: str | Path,
    preset: str,
    reference_pred: pd.Series,
    reference_component_frame: pd.DataFrame | None,
    params: CatBoostHyperParams,
    feature_blocks: Sequence[str] | None,
    folds: int,
    random_state: int,
    early_stopping_rounds: int,
    verbose: int,
    alpha_grid: Sequence[float],
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
) -> dict[str, Any]:
    """Train a specialist model on a hard cohort and scan local override alpha."""
    if folds < 2:
        raise ValueError("folds must be >= 2")

    train_df = load_csv(train_csv_path)
    normalized_blocks, _, stateful_blocks, x, y = _prepare_train_matrix(train_df, feature_blocks)
    specialist_mask = build_specialist_mask(train_df, preset).astype(bool)

    reference_by_id = pd.Series(reference_pred, copy=True)
    reference_pred = train_df[ID_COLUMN].map(reference_by_id)
    if reference_pred.isna().any():
        missing_ids = train_df.loc[reference_pred.isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference_pred missing ids from train: {missing_ids}")
    reference_pred = pd.Series(reference_pred.values, index=x.index, dtype="float64", name="reference_pred")
    component_frame = _normalize_reference_component_frame(train_df[ID_COLUMN], reference_component_frame)
    mask_rows = int(specialist_mask.sum())
    if mask_rows < _MIN_SPECIALIST_TRAIN_ROWS:
        raise ValueError(
            f"Preset '{preset}' selects only {mask_rows} rows. Minimum required: {_MIN_SPECIALIST_TRAIN_ROWS}"
        )
    if y.loc[specialist_mask].nunique() < 2:
        raise ValueError(f"Preset '{preset}' must contain both classes.")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    specialist_pred = pd.Series(index=x.index, dtype="float64", name="specialist_pred")
    fold_rows: list[dict[str, Any]] = []
    fold_iterations: list[int] = []
    disagreement_columns: list[str] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]
        x_train, x_valid = _transform_pair_with_stateful_blocks(x_train, x_valid, stateful_blocks)

        train_mask = specialist_mask.iloc[train_idx].to_numpy(dtype=bool)
        valid_mask = specialist_mask.iloc[valid_idx].to_numpy(dtype=bool)

        specialist_train_rows = int(np.sum(train_mask))
        specialist_valid_rows = int(np.sum(valid_mask))
        if specialist_train_rows < _MIN_SPECIALIST_TRAIN_ROWS:
            raise ValueError(
                f"Fold {fold_number} preset '{preset}' train rows {specialist_train_rows} below minimum."
            )
        if specialist_valid_rows < _MIN_SPECIALIST_VALID_ROWS:
            raise ValueError(
                f"Fold {fold_number} preset '{preset}' valid rows {specialist_valid_rows} below minimum."
            )
        if y_train.iloc[train_mask].nunique() < 2 or y_valid.iloc[valid_mask].nunique() < 2:
            raise ValueError(f"Fold {fold_number} preset '{preset}' lost one class in train or valid.")

        train_reference = reference_pred.iloc[train_idx]
        valid_reference = reference_pred.iloc[valid_idx]
        train_components = component_frame.iloc[train_idx] if component_frame is not None else None
        valid_components = component_frame.iloc[valid_idx] if component_frame is not None else None
        x_train, train_disagreement_columns = _append_reference_features(
            x_train,
            reference_pred=train_reference,
            reference_component_frame=train_components,
            include_logit=False,
        )
        x_valid, valid_disagreement_columns = _append_reference_features(
            x_valid,
            reference_pred=valid_reference,
            reference_component_frame=valid_components,
            include_logit=False,
        )
        if train_disagreement_columns != valid_disagreement_columns:
            raise RuntimeError("Teacher disagreement feature mismatch between train and valid folds.")
        if not disagreement_columns:
            disagreement_columns = list(train_disagreement_columns)
        cat_columns = infer_categorical_columns(x_train)

        fold_model = fit_with_validation(
            x_train=x_train.iloc[train_mask],
            y_train=y_train.iloc[train_mask],
            x_valid=x_valid.iloc[valid_mask],
            y_valid=y_valid.iloc[valid_mask],
            cat_columns=cat_columns,
            params=CatBoostHyperParams(
                iterations=params.iterations,
                learning_rate=params.learning_rate,
                depth=params.depth,
                l2_leaf_reg=params.l2_leaf_reg,
                random_seed=random_state,
                loss_function=params.loss_function,
                eval_metric=params.eval_metric,
            ),
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        valid_specialist_pred = predict_proba(fold_model, x_valid.iloc[valid_mask])
        specialist_pred.iloc[np.asarray(valid_idx)[valid_mask]] = valid_specialist_pred.values

        fold_final_iterations = best_iteration_or_default(fold_model, params.iterations)
        best_iteration = fold_model.get_best_iteration()
        if best_iteration is None or best_iteration < 0:
            best_iteration = fold_final_iterations - 1
        fold_iterations.append(fold_final_iterations)

        reference_on_mask = reference_pred.iloc[valid_idx].iloc[valid_mask]
        reference_auc = binary_auc(y_valid.iloc[valid_mask], reference_on_mask)
        specialist_auc = binary_auc(y_valid.iloc[valid_mask], valid_specialist_pred)
        fold_rows.append(
            {
                "fold": fold_number,
                "specialist_train_rows": specialist_train_rows,
                "specialist_valid_rows": specialist_valid_rows,
                "specialist_positive_rate_train": float(y_train.iloc[train_mask].mean()),
                "specialist_positive_rate_valid": float(y_valid.iloc[valid_mask].mean()),
                "reference_auc_on_mask": reference_auc,
                "specialist_auc_on_mask": specialist_auc,
                "delta_auc_on_mask": float(specialist_auc - reference_auc),
                "best_iteration": int(best_iteration),
                "final_iterations": int(fold_final_iterations),
            }
        )

    if specialist_pred.loc[specialist_mask].isna().any():
        raise RuntimeError("Specialist OOF predictions contain missing values inside the target cohort.")

    alpha_scan_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        alpha_value = float(alpha)
        candidate = reference_pred.copy()
        candidate.loc[specialist_mask] = (
            (1.0 - alpha_value) * reference_pred.loc[specialist_mask]
            + alpha_value * specialist_pred.loc[specialist_mask]
        )
        alpha_scan_rows.append(
            {
                "alpha": alpha_value,
                "oof_auc": binary_auc(y, candidate),
                "oof_auc_on_mask": binary_auc(y.loc[specialist_mask], candidate.loc[specialist_mask]),
                "reference_auc_on_mask": binary_auc(y.loc[specialist_mask], reference_pred.loc[specialist_mask]),
            }
        )

    best_alpha_row = max(alpha_scan_rows, key=lambda row: row["oof_auc"])
    best_alpha = float(best_alpha_row["alpha"])
    candidate_best = reference_pred.copy()
    candidate_best.loc[specialist_mask] = (
        (1.0 - best_alpha) * reference_pred.loc[specialist_mask]
        + best_alpha * specialist_pred.loc[specialist_mask]
    )

    final_iterations = max(int(round(float(np.mean(fold_iterations)))), 1)
    x_full = _transform_single_with_stateful_blocks(x, stateful_blocks)
    x_full, disagreement_columns_full = _append_reference_features(
        x_full,
        reference_pred=reference_pred,
        reference_component_frame=component_frame,
        include_logit=False,
    )
    if disagreement_columns and disagreement_columns_full != disagreement_columns:
        raise RuntimeError("Teacher disagreement feature mismatch between CV folds and full train.")
    if not disagreement_columns:
        disagreement_columns = list(disagreement_columns_full)
    cat_columns = infer_categorical_columns(x_full)
    full_model = fit_full_train(
        x_train=x_full.loc[specialist_mask],
        y_train=y.loc[specialist_mask],
        cat_columns=cat_columns,
        params=CatBoostHyperParams(
            iterations=final_iterations,
            learning_rate=params.learning_rate,
            depth=params.depth,
            l2_leaf_reg=params.l2_leaf_reg,
            random_seed=params.random_seed,
            loss_function=params.loss_function,
            eval_metric=params.eval_metric,
        ),
        verbose=verbose,
    )
    save_model(full_model, model_path)

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "specialist_mask": specialist_mask.astype("int8").values,
            "reference_pred": reference_pred.values,
            "specialist_pred": specialist_pred.values,
            "candidate_pred": candidate_best.values,
        }
    )
    oof_out_path = ensure_parent_dir(oof_path)
    oof_output.to_csv(oof_out_path, index=False)

    metrics: dict[str, Any] = {
        "preset": preset,
        "preset_description": SPECIALIST_PRESETS[preset],
        "train_rows": int(len(train_df)),
        "specialist_rows": mask_rows,
        "specialist_row_share": float(mask_rows / max(len(train_df), 1)),
        "specialist_positive_rate": float(y.loc[specialist_mask].mean()),
        "feature_count": int(x_full.shape[1]),
        "feature_columns": list(x_full.columns),
        "feature_blocks": list(normalized_blocks),
        "teacher_disagreement_columns": disagreement_columns,
        "categorical_columns": cat_columns,
        "cv_folds": int(folds),
        "cv_fold_metrics": fold_rows,
        "fold_final_iterations": [int(value) for value in fold_iterations],
        "final_iterations": int(final_iterations),
        "alpha_scan": alpha_scan_rows,
        "best_alpha": best_alpha,
        "reference_oof_auc": binary_auc(y, reference_pred),
        "reference_oof_auc_on_mask": binary_auc(y.loc[specialist_mask], reference_pred.loc[specialist_mask]),
        "specialist_oof_auc_on_mask": binary_auc(y.loc[specialist_mask], specialist_pred.loc[specialist_mask]),
        "candidate_oof_auc": binary_auc(y, candidate_best),
        "candidate_oof_auc_on_mask": binary_auc(y.loc[specialist_mask], candidate_best.loc[specialist_mask]),
        "delta_vs_reference_oof_auc": float(binary_auc(y, candidate_best) - binary_auc(y, reference_pred)),
        "model_path": str(model_path),
        "oof_path": str(oof_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
    }
    write_json(metrics_path, metrics)
    return metrics


def run_residual_reranker_cv(
    *,
    train_csv_path: str | Path,
    preset: str,
    reference_pred: pd.Series,
    reference_component_frame: pd.DataFrame | None,
    params: CatBoostHyperParams,
    feature_blocks: Sequence[str] | None,
    folds: int,
    random_state: int,
    early_stopping_rounds: int,
    verbose: int,
    alpha_grid: Sequence[float],
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
) -> dict[str, Any]:
    """Train a local residual reranker over the incumbent score inside a hard cohort."""
    if folds < 2:
        raise ValueError("folds must be >= 2")

    train_df = load_csv(train_csv_path)
    normalized_blocks, _, stateful_blocks, x, y = _prepare_train_matrix(train_df, feature_blocks)
    specialist_mask = build_specialist_mask(train_df, preset).astype(bool)

    reference_by_id = pd.Series(reference_pred, copy=True)
    reference_pred = train_df[ID_COLUMN].map(reference_by_id)
    if reference_pred.isna().any():
        missing_ids = train_df.loc[reference_pred.isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference_pred missing ids from train: {missing_ids}")
    reference_pred = pd.Series(reference_pred.values, index=x.index, dtype="float64", name="reference_pred")
    component_frame = _normalize_reference_component_frame(train_df[ID_COLUMN], reference_component_frame)

    mask_rows = int(specialist_mask.sum())
    if mask_rows < _MIN_SPECIALIST_TRAIN_ROWS:
        raise ValueError(
            f"Preset '{preset}' selects only {mask_rows} rows. Minimum required: {_MIN_SPECIALIST_TRAIN_ROWS}"
        )
    if y.loc[specialist_mask].nunique() < 2:
        raise ValueError(f"Preset '{preset}' must contain both classes.")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    residual_pred = pd.Series(index=x.index, dtype="float64", name="residual_pred")
    fold_rows: list[dict[str, Any]] = []
    fold_iterations: list[int] = []
    disagreement_columns: list[str] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]
        x_train, x_valid = _transform_pair_with_stateful_blocks(x_train, x_valid, stateful_blocks)

        train_mask = specialist_mask.iloc[train_idx].to_numpy(dtype=bool)
        valid_mask = specialist_mask.iloc[valid_idx].to_numpy(dtype=bool)

        specialist_train_rows = int(np.sum(train_mask))
        specialist_valid_rows = int(np.sum(valid_mask))
        if specialist_train_rows < _MIN_SPECIALIST_TRAIN_ROWS:
            raise ValueError(
                f"Fold {fold_number} preset '{preset}' train rows {specialist_train_rows} below minimum."
            )
        if specialist_valid_rows < _MIN_SPECIALIST_VALID_ROWS:
            raise ValueError(
                f"Fold {fold_number} preset '{preset}' valid rows {specialist_valid_rows} below minimum."
            )
        if y_train.iloc[train_mask].nunique() < 2 or y_valid.iloc[valid_mask].nunique() < 2:
            raise ValueError(f"Fold {fold_number} preset '{preset}' lost one class in train or valid.")

        reference_train_mask = reference_pred.iloc[train_idx].iloc[train_mask]
        reference_valid_mask = reference_pred.iloc[valid_idx].iloc[valid_mask]
        train_reference = reference_pred.iloc[train_idx]
        valid_reference = reference_pred.iloc[valid_idx]
        train_components = component_frame.iloc[train_idx] if component_frame is not None else None
        valid_components = component_frame.iloc[valid_idx] if component_frame is not None else None
        x_train, train_disagreement_columns = _append_reference_features(
            x_train,
            reference_pred=train_reference,
            reference_component_frame=train_components,
            include_logit=True,
        )
        x_valid, valid_disagreement_columns = _append_reference_features(
            x_valid,
            reference_pred=valid_reference,
            reference_component_frame=valid_components,
            include_logit=True,
        )
        if train_disagreement_columns != valid_disagreement_columns:
            raise RuntimeError("Teacher disagreement feature mismatch between train and valid folds.")
        if not disagreement_columns:
            disagreement_columns = list(train_disagreement_columns)
        cat_columns = infer_categorical_columns(x_train)
        y_train_residual = y_train.iloc[train_mask].astype("float64") - reference_train_mask.astype("float64")

        fold_model = CatBoostRegressor(
            iterations=params.iterations,
            learning_rate=params.learning_rate,
            depth=params.depth,
            l2_leaf_reg=params.l2_leaf_reg,
            random_seed=random_state,
            loss_function="RMSE",
            eval_metric="RMSE",
            verbose=verbose,
        )
        fold_model.fit(
            x_train.iloc[train_mask],
            y_train_residual,
            cat_features=cat_columns,
            eval_set=(
                x_valid.iloc[valid_mask],
                y_valid.iloc[valid_mask].astype("float64") - reference_valid_mask.astype("float64"),
            ),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        valid_residual_pred = pd.Series(
            fold_model.predict(x_valid.iloc[valid_mask]),
            index=x_valid.iloc[valid_mask].index,
            dtype="float64",
        )
        residual_pred.iloc[np.asarray(valid_idx)[valid_mask]] = valid_residual_pred.values

        fold_final_iterations = best_iteration_or_default(fold_model, params.iterations)
        best_iteration = fold_model.get_best_iteration()
        if best_iteration is None or best_iteration < 0:
            best_iteration = fold_final_iterations - 1
        fold_iterations.append(fold_final_iterations)

        reference_auc = binary_auc(y_valid.iloc[valid_mask], reference_valid_mask)
        reranked_valid = _clip_probability(reference_valid_mask + valid_residual_pred)
        reranked_auc = binary_auc(y_valid.iloc[valid_mask], reranked_valid)
        fold_rows.append(
            {
                "fold": fold_number,
                "specialist_train_rows": specialist_train_rows,
                "specialist_valid_rows": specialist_valid_rows,
                "specialist_positive_rate_train": float(y_train.iloc[train_mask].mean()),
                "specialist_positive_rate_valid": float(y_valid.iloc[valid_mask].mean()),
                "reference_auc_on_mask": reference_auc,
                "reranked_auc_on_mask": reranked_auc,
                "delta_auc_on_mask": float(reranked_auc - reference_auc),
                "best_iteration": int(best_iteration),
                "final_iterations": int(fold_final_iterations),
            }
        )

    if residual_pred.loc[specialist_mask].isna().any():
        raise RuntimeError("Residual OOF predictions contain missing values inside the target cohort.")

    alpha_scan_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        alpha_value = float(alpha)
        candidate = reference_pred.copy()
        candidate.loc[specialist_mask] = _clip_probability(
            reference_pred.loc[specialist_mask] + alpha_value * residual_pred.loc[specialist_mask]
        ).values
        alpha_scan_rows.append(
            {
                "alpha": alpha_value,
                "oof_auc": binary_auc(y, candidate),
                "oof_auc_on_mask": binary_auc(y.loc[specialist_mask], candidate.loc[specialist_mask]),
                "reference_auc_on_mask": binary_auc(y.loc[specialist_mask], reference_pred.loc[specialist_mask]),
            }
        )

    best_alpha_row = max(alpha_scan_rows, key=lambda row: row["oof_auc"])
    best_alpha = float(best_alpha_row["alpha"])
    candidate_best = reference_pred.copy()
    candidate_best.loc[specialist_mask] = _clip_probability(
        reference_pred.loc[specialist_mask] + best_alpha * residual_pred.loc[specialist_mask]
    ).values

    final_iterations = max(int(round(float(np.mean(fold_iterations)))), 1)
    x_full = _transform_single_with_stateful_blocks(x, stateful_blocks)
    x_full, disagreement_columns_full = _append_reference_features(
        x_full,
        reference_pred=reference_pred,
        reference_component_frame=component_frame,
        include_logit=True,
    )
    if disagreement_columns and disagreement_columns_full != disagreement_columns:
        raise RuntimeError("Teacher disagreement feature mismatch between CV folds and full train.")
    if not disagreement_columns:
        disagreement_columns = list(disagreement_columns_full)
    cat_columns = infer_categorical_columns(x_full)
    full_model = CatBoostRegressor(
        iterations=final_iterations,
        learning_rate=params.learning_rate,
        depth=params.depth,
        l2_leaf_reg=params.l2_leaf_reg,
        random_seed=params.random_seed,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=verbose,
    )
    full_model.fit(
        x_full.loc[specialist_mask],
        (y.loc[specialist_mask].astype("float64") - reference_pred.loc[specialist_mask].astype("float64")),
        cat_features=cat_columns,
        verbose=verbose,
    )
    save_model(full_model, model_path)

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "specialist_mask": specialist_mask.astype("int8").values,
            "reference_pred": reference_pred.values,
            "residual_pred": residual_pred.values,
            "candidate_pred": candidate_best.values,
        }
    )
    oof_out_path = ensure_parent_dir(oof_path)
    oof_output.to_csv(oof_out_path, index=False)

    metrics: dict[str, Any] = {
        "preset": preset,
        "preset_description": SPECIALIST_PRESETS[preset],
        "approach": RESIDUAL,
        "train_rows": int(len(train_df)),
        "specialist_rows": mask_rows,
        "specialist_row_share": float(mask_rows / max(len(train_df), 1)),
        "specialist_positive_rate": float(y.loc[specialist_mask].mean()),
        "feature_count": int(x_full.shape[1]),
        "feature_columns": list(x_full.columns),
        "feature_blocks": list(normalized_blocks),
        "teacher_disagreement_columns": disagreement_columns,
        "categorical_columns": cat_columns,
        "cv_folds": int(folds),
        "cv_fold_metrics": fold_rows,
        "fold_final_iterations": [int(value) for value in fold_iterations],
        "final_iterations": int(final_iterations),
        "alpha_scan": alpha_scan_rows,
        "best_alpha": best_alpha,
        "reference_oof_auc": binary_auc(y, reference_pred),
        "reference_oof_auc_on_mask": binary_auc(y.loc[specialist_mask], reference_pred.loc[specialist_mask]),
        "reranked_oof_auc_on_mask": binary_auc(y.loc[specialist_mask], candidate_best.loc[specialist_mask]),
        "candidate_oof_auc": binary_auc(y, candidate_best),
        "candidate_oof_auc_on_mask": binary_auc(y.loc[specialist_mask], candidate_best.loc[specialist_mask]),
        "delta_vs_reference_oof_auc": float(binary_auc(y, candidate_best) - binary_auc(y, reference_pred)),
        "model_path": str(model_path),
        "oof_path": str(oof_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
    }
    write_json(metrics_path, metrics)
    return metrics


def run_local_calibrator_cv(
    *,
    train_csv_path: str | Path,
    preset: str,
    reference_pred: pd.Series,
    method: str,
    folds: int,
    random_state: int,
    alpha_grid: Sequence[float],
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
) -> dict[str, Any]:
    """Fit a local calibrator on incumbent scores inside a hard cohort."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if str(method).strip().lower() not in CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method '{method}'. Available: {CALIBRATION_METHODS}")

    train_df = load_csv(train_csv_path)
    y = normalize_binary_target(train_df[TARGET_COLUMN])
    specialist_mask = build_specialist_mask(train_df, preset).astype(bool)

    reference_by_id = pd.Series(reference_pred, copy=True)
    reference_pred = train_df[ID_COLUMN].map(reference_by_id)
    if reference_pred.isna().any():
        missing_ids = train_df.loc[reference_pred.isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference_pred missing ids from train: {missing_ids}")
    reference_pred = pd.Series(reference_pred.values, index=train_df.index, dtype="float64", name="reference_pred")

    mask_rows = int(specialist_mask.sum())
    if mask_rows < _MIN_SPECIALIST_TRAIN_ROWS:
        raise ValueError(
            f"Preset '{preset}' selects only {mask_rows} rows. Minimum required: {_MIN_SPECIALIST_TRAIN_ROWS}"
        )
    if y.loc[specialist_mask].nunique() < 2:
        raise ValueError(f"Preset '{preset}' must contain both classes.")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    calibrated_pred = pd.Series(index=train_df.index, dtype="float64", name="calibrated_pred")
    fold_rows: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(train_df, y), start=1):
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        train_mask = specialist_mask.iloc[train_idx].to_numpy(dtype=bool)
        valid_mask = specialist_mask.iloc[valid_idx].to_numpy(dtype=bool)

        specialist_train_rows = int(np.sum(train_mask))
        specialist_valid_rows = int(np.sum(valid_mask))
        if specialist_train_rows < _MIN_SPECIALIST_TRAIN_ROWS:
            raise ValueError(
                f"Fold {fold_number} preset '{preset}' train rows {specialist_train_rows} below minimum."
            )
        if specialist_valid_rows < _MIN_SPECIALIST_VALID_ROWS:
            raise ValueError(
                f"Fold {fold_number} preset '{preset}' valid rows {specialist_valid_rows} below minimum."
            )
        if y_train.iloc[train_mask].nunique() < 2 or y_valid.iloc[valid_mask].nunique() < 2:
            raise ValueError(f"Fold {fold_number} preset '{preset}' lost one class in train or valid.")

        train_pred_mask = reference_pred.iloc[train_idx].iloc[train_mask]
        valid_pred_mask = reference_pred.iloc[valid_idx].iloc[valid_mask]
        bundle = _fit_local_calibrator(
            method=method,
            pred=train_pred_mask,
            y_true=y_train.iloc[train_mask],
        )
        valid_calibrated_pred = _predict_local_calibrator(bundle, valid_pred_mask)
        calibrated_pred.iloc[np.asarray(valid_idx)[valid_mask]] = valid_calibrated_pred.values

        reference_auc = binary_auc(y_valid.iloc[valid_mask], valid_pred_mask)
        calibrated_auc = binary_auc(y_valid.iloc[valid_mask], valid_calibrated_pred)
        fold_rows.append(
            {
                "fold": fold_number,
                "specialist_train_rows": specialist_train_rows,
                "specialist_valid_rows": specialist_valid_rows,
                "specialist_positive_rate_train": float(y_train.iloc[train_mask].mean()),
                "specialist_positive_rate_valid": float(y_valid.iloc[valid_mask].mean()),
                "reference_auc_on_mask": reference_auc,
                "calibrated_auc_on_mask": calibrated_auc,
                "delta_auc_on_mask": float(calibrated_auc - reference_auc),
            }
        )

    if calibrated_pred.loc[specialist_mask].isna().any():
        raise RuntimeError("Calibrated OOF predictions contain missing values inside the target cohort.")

    alpha_scan_rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        alpha_value = float(alpha)
        candidate = reference_pred.copy()
        candidate.loc[specialist_mask] = (
            (1.0 - alpha_value) * reference_pred.loc[specialist_mask]
            + alpha_value * calibrated_pred.loc[specialist_mask]
        )
        alpha_scan_rows.append(
            {
                "alpha": alpha_value,
                "oof_auc": binary_auc(y, candidate),
                "oof_auc_on_mask": binary_auc(y.loc[specialist_mask], candidate.loc[specialist_mask]),
                "reference_auc_on_mask": binary_auc(y.loc[specialist_mask], reference_pred.loc[specialist_mask]),
            }
        )

    best_alpha_row = max(alpha_scan_rows, key=lambda row: row["oof_auc"])
    best_alpha = float(best_alpha_row["alpha"])
    candidate_best = reference_pred.copy()
    candidate_best.loc[specialist_mask] = (
        (1.0 - best_alpha) * reference_pred.loc[specialist_mask]
        + best_alpha * calibrated_pred.loc[specialist_mask]
    )

    full_bundle = _fit_local_calibrator(
        method=method,
        pred=reference_pred.loc[specialist_mask],
        y_true=y.loc[specialist_mask],
    )
    _save_pickle(
        model_path,
        {
            "preset": preset,
            "method": str(method).strip().lower(),
            "alpha": best_alpha,
            "calibrator": full_bundle,
        },
    )

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "specialist_mask": specialist_mask.astype("int8").values,
            "reference_pred": reference_pred.values,
            "calibrated_pred": calibrated_pred.values,
            "candidate_pred": candidate_best.values,
        }
    )
    oof_out_path = ensure_parent_dir(oof_path)
    oof_output.to_csv(oof_out_path, index=False)

    metrics: dict[str, Any] = {
        "preset": preset,
        "preset_description": SPECIALIST_PRESETS[preset],
        "calibration_method": str(method).strip().lower(),
        "train_rows": int(len(train_df)),
        "specialist_rows": mask_rows,
        "specialist_row_share": float(mask_rows / max(len(train_df), 1)),
        "specialist_positive_rate": float(y.loc[specialist_mask].mean()),
        "cv_folds": int(folds),
        "cv_fold_metrics": fold_rows,
        "alpha_scan": alpha_scan_rows,
        "best_alpha": best_alpha,
        "reference_oof_auc": binary_auc(y, reference_pred),
        "reference_oof_auc_on_mask": binary_auc(y.loc[specialist_mask], reference_pred.loc[specialist_mask]),
        "calibrated_oof_auc_on_mask": binary_auc(y.loc[specialist_mask], calibrated_pred.loc[specialist_mask]),
        "candidate_oof_auc": binary_auc(y, candidate_best),
        "candidate_oof_auc_on_mask": binary_auc(y.loc[specialist_mask], candidate_best.loc[specialist_mask]),
        "delta_vs_reference_oof_auc": float(binary_auc(y, candidate_best) - binary_auc(y, reference_pred)),
        "model_path": str(model_path),
        "oof_path": str(oof_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
    }
    write_json(metrics_path, metrics)
    return metrics


def make_specialist_override_prediction(
    *,
    test_csv_path: str | Path,
    reference_submission_path: str | Path,
    reference_component_frame: pd.DataFrame | None = None,
    train_csv_path: str | Path | None = None,
    model_path: str | Path,
    preset: str,
    feature_blocks: Sequence[str] | None,
    alpha: float,
) -> pd.DataFrame:
    """Apply a trained specialist override on top of a reference submission."""
    if float(alpha) < 0.0 or float(alpha) > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    test_df = load_csv(test_csv_path)
    reference_submission = pd.read_csv(reference_submission_path)
    if ID_COLUMN not in reference_submission.columns or "Churn" not in reference_submission.columns:
        raise ValueError(f"{reference_submission_path} must contain columns '{ID_COLUMN}' and 'Churn'.")

    _, x_test = _prepare_specialist_test_matrix(
        test_df=test_df,
        feature_blocks=feature_blocks,
        train_csv_path=train_csv_path,
    )
    specialist_mask = build_specialist_mask(test_df, preset).astype(bool)
    specialist_model = load_model(model_path)
    output = test_df[[ID_COLUMN]].merge(reference_submission[[ID_COLUMN, "Churn"]], how="left", on=ID_COLUMN, validate="one_to_one")
    if output["Churn"].isna().any():
        missing_ids = output.loc[output["Churn"].isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference submission missing ids from test: {missing_ids}")
    component_frame = _normalize_reference_component_frame(test_df[ID_COLUMN], reference_component_frame)
    x_test, _ = _append_reference_features(
        x_test,
        reference_pred=pd.Series(output["Churn"].astype("float64").values, index=x_test.index, dtype="float64"),
        reference_component_frame=component_frame,
        include_logit=False,
    )
    specialist_pred = pd.Series(np.nan, index=x_test.index, dtype="float64")
    expected_columns = list(getattr(specialist_model, "feature_names_", []))
    if expected_columns:
        missing = [column for column in expected_columns if column not in x_test.columns]
        if missing:
            raise ValueError(f"Missing inference columns for specialist model: {missing}")
        x_test = x_test.loc[:, expected_columns]
    specialist_pred.loc[specialist_mask] = predict_proba(specialist_model, x_test.loc[specialist_mask]).values
    output["specialist_mask"] = specialist_mask.astype("int8").values
    output["specialist_pred"] = specialist_pred.values
    output.loc[specialist_mask, "Churn"] = (
        (1.0 - float(alpha)) * output.loc[specialist_mask, "Churn"].astype("float64")
        + float(alpha) * specialist_pred.loc[specialist_mask]
    )
    return output


def _load_reference_submission_frame(
    *,
    test_df: pd.DataFrame,
    reference_submission_path: str | Path,
) -> pd.DataFrame:
    """Load and align a reference submission against the test ids."""
    reference_submission = pd.read_csv(reference_submission_path)
    if ID_COLUMN not in reference_submission.columns or "Churn" not in reference_submission.columns:
        raise ValueError(f"{reference_submission_path} must contain columns '{ID_COLUMN}' and 'Churn'.")

    output = test_df[[ID_COLUMN]].merge(
        reference_submission[[ID_COLUMN, "Churn"]],
        how="left",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if output["Churn"].isna().any():
        missing_ids = output.loc[output["Churn"].isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference submission missing ids from test: {missing_ids}")
    return output


def _predict_catboost_ensemble_frame(
    *,
    test_df: pd.DataFrame,
    model_paths: Sequence[str | Path],
    feature_blocks: Sequence[str] | None,
    train_csv_path: str | Path | None = None,
) -> pd.Series:
    """Predict average CatBoost probabilities for a list of models on test rows."""
    paths = [str(path) for path in model_paths if str(path).strip()]
    if not paths:
        raise ValueError("model_paths must contain at least one model path")

    _, x_test = _prepare_specialist_test_matrix(
        test_df=test_df,
        feature_blocks=feature_blocks,
        train_csv_path=train_csv_path,
    )
    total = np.zeros(len(test_df), dtype="float64")

    for model_path in paths:
        model = load_model(model_path)
        expected_columns = list(getattr(model, "feature_names_", []))
        x_model = x_test.loc[:, expected_columns] if expected_columns else x_test
        total += predict_proba(model, x_model).to_numpy(dtype="float64")

    return pd.Series(total / len(paths), index=test_df.index, dtype="float64")


def build_teacher_reference_component_frame(
    *,
    test_csv_path: str | Path,
    train_csv_path: str | Path | None = None,
    cb_model_paths: Sequence[str | Path],
    cb_feature_blocks: Sequence[str] | None,
    r_model_path: str | Path,
    r_feature_blocks: Sequence[str] | None,
    cbxgblgb_submission_path: str | Path,
    cbxgblgb_weights: dict[str, float],
    cbr_submission_path: str | Path,
    cbr_weights: dict[str, float],
    cbrv_submission_path: str | Path,
    cbrv_weights: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct incumbent teacher components from validated blend submissions."""
    required_cbxgblgb = {"cb", "xgb", "lgb"}
    required_cbr = {"cb", "xgb", "lgb", "r"}
    required_cbrv = {"cb", "xgb", "lgb", "r", "rv"}
    if not required_cbxgblgb.issubset(cbxgblgb_weights):
        raise ValueError(f"cbxgblgb_weights must contain {sorted(required_cbxgblgb)}")
    if not required_cbr.issubset(cbr_weights):
        raise ValueError(f"cbr_weights must contain {sorted(required_cbr)}")
    if not required_cbrv.issubset(cbrv_weights):
        raise ValueError(f"cbrv_weights must contain {sorted(required_cbrv)}")

    test_df = load_csv(test_csv_path)
    test_ids = test_df[ID_COLUMN].copy()

    cb_pred = _predict_catboost_ensemble_frame(
        test_df=test_df,
        model_paths=cb_model_paths,
        feature_blocks=cb_feature_blocks,
        train_csv_path=train_csv_path,
    ).to_numpy(dtype="float64")
    r_pred = _predict_catboost_ensemble_frame(
        test_df=test_df,
        model_paths=[r_model_path],
        feature_blocks=r_feature_blocks,
        train_csv_path=train_csv_path,
    ).to_numpy(dtype="float64")

    cbxgblgb_frame = _load_reference_submission_frame(
        test_df=test_df,
        reference_submission_path=cbxgblgb_submission_path,
    )
    cbr_frame = _load_reference_submission_frame(
        test_df=test_df,
        reference_submission_path=cbr_submission_path,
    )
    cbrv_frame = _load_reference_submission_frame(
        test_df=test_df,
        reference_submission_path=cbrv_submission_path,
    )

    a0 = float(cbxgblgb_weights["xgb"])
    b0 = float(cbxgblgb_weights["lgb"])
    a1 = float(cbr_weights["xgb"])
    b1 = float(cbr_weights["lgb"])
    determinant = a0 * b1 - a1 * b0
    if abs(determinant) < 1e-12:
        raise ValueError(f"Blend system is ill-conditioned for xgb/lgb reconstruction: det={determinant}")

    rhs0 = cbxgblgb_frame["Churn"].astype("float64").to_numpy(dtype="float64") - float(cbxgblgb_weights["cb"]) * cb_pred
    rhs1 = (
        cbr_frame["Churn"].astype("float64").to_numpy(dtype="float64")
        - float(cbr_weights["cb"]) * cb_pred
        - float(cbr_weights["r"]) * r_pred
    )

    xgb_pred = (rhs0 * b1 - rhs1 * b0) / determinant
    lgb_pred = (a0 * rhs1 - a1 * rhs0) / determinant
    rv_pred = (
        cbrv_frame["Churn"].astype("float64").to_numpy(dtype="float64")
        - float(cbrv_weights["cb"]) * cb_pred
        - float(cbrv_weights["xgb"]) * xgb_pred
        - float(cbrv_weights["lgb"]) * lgb_pred
        - float(cbrv_weights["r"]) * r_pred
    ) / float(cbrv_weights["rv"])

    component_frame = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            "pred_cb": cb_pred,
            "pred_xgb": xgb_pred,
            "pred_lgb": lgb_pred,
            "pred_r": r_pred,
            "pred_rv": rv_pred,
        }
    )

    reconstructed_cbxgblgb = (
        float(cbxgblgb_weights["cb"]) * component_frame["pred_cb"]
        + float(cbxgblgb_weights["xgb"]) * component_frame["pred_xgb"]
        + float(cbxgblgb_weights["lgb"]) * component_frame["pred_lgb"]
    )
    reconstructed_cbr = (
        float(cbr_weights["cb"]) * component_frame["pred_cb"]
        + float(cbr_weights["xgb"]) * component_frame["pred_xgb"]
        + float(cbr_weights["lgb"]) * component_frame["pred_lgb"]
        + float(cbr_weights["r"]) * component_frame["pred_r"]
    )
    reconstructed_cbrv = (
        float(cbrv_weights["cb"]) * component_frame["pred_cb"]
        + float(cbrv_weights["xgb"]) * component_frame["pred_xgb"]
        + float(cbrv_weights["lgb"]) * component_frame["pred_lgb"]
        + float(cbrv_weights["r"]) * component_frame["pred_r"]
        + float(cbrv_weights["rv"]) * component_frame["pred_rv"]
    )

    report: dict[str, Any] = {
        "test_csv_path": str(test_csv_path),
        "train_csv_path": None if train_csv_path is None else str(train_csv_path),
        "cb_model_paths": [str(path) for path in cb_model_paths],
        "cb_feature_blocks": list(normalize_feature_blocks(cb_feature_blocks)),
        "r_model_path": str(r_model_path),
        "r_feature_blocks": list(normalize_feature_blocks(r_feature_blocks)),
        "cbxgblgb_submission_path": str(cbxgblgb_submission_path),
        "cbr_submission_path": str(cbr_submission_path),
        "cbrv_submission_path": str(cbrv_submission_path),
        "rows": int(len(component_frame)),
        "component_reconstruction": {
            "cbxgblgb_max_abs_err": float(
                np.max(np.abs(reconstructed_cbxgblgb.to_numpy(dtype="float64") - cbxgblgb_frame["Churn"].astype("float64").to_numpy(dtype="float64")))
            ),
            "cbr_max_abs_err": float(
                np.max(np.abs(reconstructed_cbr.to_numpy(dtype="float64") - cbr_frame["Churn"].astype("float64").to_numpy(dtype="float64")))
            ),
            "cbrv_max_abs_err": float(
                np.max(np.abs(reconstructed_cbrv.to_numpy(dtype="float64") - cbrv_frame["Churn"].astype("float64").to_numpy(dtype="float64")))
            ),
        },
        "component_ranges": {
            column: {
                "min": float(component_frame[column].min()),
                "max": float(component_frame[column].max()),
            }
            for column in ["pred_cb", "pred_xgb", "pred_lgb", "pred_r", "pred_rv"]
        },
    }
    return component_frame, report


def _apply_residual_reranker_frame(
    *,
    test_df: pd.DataFrame,
    reference_frame: pd.DataFrame,
    reference_component_frame: pd.DataFrame | None,
    train_csv_path: str | Path | None,
    model_path: str | Path,
    preset: str,
    feature_blocks: Sequence[str] | None,
    alpha: float,
) -> pd.DataFrame:
    """Apply a residual reranker over an in-memory reference submission frame."""
    if float(alpha) < 0.0 or float(alpha) > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if ID_COLUMN not in reference_frame.columns or "Churn" not in reference_frame.columns:
        raise ValueError("reference_frame must contain 'id' and 'Churn' columns.")

    _, x_test = _prepare_specialist_test_matrix(
        test_df=test_df,
        feature_blocks=feature_blocks,
        train_csv_path=train_csv_path,
    )
    component_frame = _normalize_reference_component_frame(test_df[ID_COLUMN], reference_component_frame)
    specialist_mask = build_specialist_mask(test_df, preset).astype(bool)
    output = test_df[[ID_COLUMN]].merge(
        reference_frame[[ID_COLUMN, "Churn"]],
        how="left",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if output["Churn"].isna().any():
        missing_ids = output.loc[output["Churn"].isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference frame missing ids from test: {missing_ids}")

    x_test, _ = _append_reference_features(
        x_test,
        reference_pred=pd.Series(output["Churn"].astype("float64").values, index=x_test.index, dtype="float64"),
        reference_component_frame=component_frame,
        include_logit=True,
    )

    residual_model = CatBoostRegressor()
    residual_model.load_model(str(model_path))
    expected_columns = list(getattr(residual_model, "feature_names_", []))
    if expected_columns:
        missing = [column for column in expected_columns if column not in x_test.columns]
        if missing:
            raise ValueError(f"Missing inference columns for residual reranker: {missing}")
        x_test = x_test.loc[:, expected_columns]

    residual_pred = pd.Series(np.nan, index=x_test.index, dtype="float64")
    residual_pred.loc[specialist_mask] = residual_model.predict(x_test.loc[specialist_mask])
    output["specialist_mask"] = specialist_mask.astype("int8").values
    output["residual_pred"] = residual_pred.values
    output.loc[specialist_mask, "Churn"] = _clip_probability(
        output.loc[specialist_mask, "Churn"].astype("float64") + float(alpha) * residual_pred.loc[specialist_mask]
    ).values
    return output


def make_residual_reranker_prediction(
    *,
    test_csv_path: str | Path,
    reference_submission_path: str | Path,
    reference_component_frame: pd.DataFrame | None = None,
    train_csv_path: str | Path | None = None,
    model_path: str | Path,
    preset: str,
    feature_blocks: Sequence[str] | None,
    alpha: float,
) -> pd.DataFrame:
    """Apply a trained residual reranker on top of a reference submission."""
    test_df = load_csv(test_csv_path)
    reference_frame = _load_reference_submission_frame(
        test_df=test_df,
        reference_submission_path=reference_submission_path,
    )
    return _apply_residual_reranker_frame(
        test_df=test_df,
        reference_frame=reference_frame,
        reference_component_frame=reference_component_frame,
        train_csv_path=train_csv_path,
        model_path=model_path,
        preset=preset,
        feature_blocks=feature_blocks,
        alpha=alpha,
    )


def make_residual_reranker_chain_submission(
    *,
    test_csv_path: str | Path,
    reference_submission_path: str | Path,
    reference_component_frame: pd.DataFrame | None = None,
    train_csv_path: str | Path | None = None,
    reference_mode: str = "previous",
    steps: Sequence[dict[str, Any]],
    output_csv_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Apply a sequence of residual rerankers over a reference submission and write artifacts."""
    if not steps:
        raise ValueError("steps must contain at least one residual reranker spec.")
    if reference_mode not in {"previous", "base"}:
        raise ValueError(f"reference_mode must be 'previous' or 'base', got {reference_mode!r}")
    if reference_component_frame is not None and reference_mode != "base":
        raise ValueError("teacher-aware chaining requires reference_mode='base' when reference_component_frame is provided")

    test_df = load_csv(test_csv_path)
    current_output = _load_reference_submission_frame(
        test_df=test_df,
        reference_submission_path=reference_submission_path,
    )
    base_output = current_output.copy()
    base_pred = current_output["Churn"].astype("float64").copy()
    step_rows: list[dict[str, Any]] = []

    for index, raw_step in enumerate(steps, start=1):
        preset = str(raw_step["preset"])
        model_path = str(raw_step["model_path"])
        alpha = float(raw_step["alpha"])
        feature_blocks = normalize_feature_blocks(raw_step.get("feature_blocks"))
        reference_frame = current_output[[ID_COLUMN, "Churn"]].copy() if reference_mode == "previous" else base_output[[ID_COLUMN, "Churn"]].copy()

        step_output = _apply_residual_reranker_frame(
            test_df=test_df,
            reference_frame=reference_frame,
            reference_component_frame=reference_component_frame,
            train_csv_path=train_csv_path,
            model_path=model_path,
            preset=preset,
            feature_blocks=feature_blocks,
            alpha=alpha,
        )
        if reference_mode == "base":
            next_output = current_output[[ID_COLUMN, "Churn"]].copy()
            step_mask = step_output["specialist_mask"].astype(bool)
            next_output.loc[step_mask, "Churn"] = step_output.loc[step_mask, "Churn"].astype("float64").values
            next_output["specialist_mask"] = step_output["specialist_mask"].astype("int8").values
            if "residual_pred" in step_output.columns:
                next_output["residual_pred"] = step_output["residual_pred"].values
            if "specialist_pred" in step_output.columns:
                next_output["specialist_pred"] = step_output["specialist_pred"].values
        else:
            next_output = step_output

        delta = next_output["Churn"].astype("float64") - current_output["Churn"].astype("float64")
        changed_mask = delta.abs().gt(1e-15)
        step_rows.append(
            {
                "step": int(index),
                "preset": preset,
                "model_path": model_path,
                "alpha": alpha,
                "feature_blocks": list(feature_blocks),
                "mask_rows": int(next_output["specialist_mask"].sum()),
                "changed_rows": int(changed_mask.sum()),
                "mae_shift_vs_previous": float(delta.abs().mean()),
                "max_abs_shift_vs_previous": float(delta.abs().max()),
            }
        )
        current_output = next_output[[ID_COLUMN, "Churn"]].copy()

    out_path = ensure_parent_dir(output_csv_path)
    current_output.to_csv(out_path, index=False)

    final_delta = current_output["Churn"].astype("float64") - base_pred
    report: dict[str, Any] = {
        "test_csv_path": str(test_csv_path),
        "train_csv_path": None if train_csv_path is None else str(train_csv_path),
        "reference_submission_path": str(reference_submission_path),
        "reference_mode": reference_mode,
        "output_csv_path": str(output_csv_path),
        "report_path": str(report_path),
        "steps": step_rows,
        "test_rows": int(len(current_output)),
        "changed_rows_vs_base": int(final_delta.abs().gt(1e-15).sum()),
        "mae_vs_base": float(final_delta.abs().mean()),
        "max_abs_shift_vs_base": float(final_delta.abs().max()),
        "min_pred": float(current_output["Churn"].min()),
        "max_pred": float(current_output["Churn"].max()),
    }
    write_json(report_path, report)
    return report


def make_local_calibrated_prediction(
    *,
    test_csv_path: str | Path,
    reference_submission_path: str | Path,
    model_path: str | Path,
) -> pd.DataFrame:
    """Apply a saved local calibrator bundle on top of a reference submission."""
    bundle = _load_pickle(model_path)
    preset = str(bundle["preset"])
    alpha = float(bundle["alpha"])
    calibrator = dict(bundle["calibrator"])

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    test_df = load_csv(test_csv_path)
    reference_submission = pd.read_csv(reference_submission_path)
    if ID_COLUMN not in reference_submission.columns or "Churn" not in reference_submission.columns:
        raise ValueError(f"{reference_submission_path} must contain columns '{ID_COLUMN}' and 'Churn'.")

    output = test_df[[ID_COLUMN]].merge(reference_submission[[ID_COLUMN, "Churn"]], how="left", on=ID_COLUMN, validate="one_to_one")
    if output["Churn"].isna().any():
        missing_ids = output.loc[output["Churn"].isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"reference submission missing ids from test: {missing_ids}")

    specialist_mask = build_specialist_mask(test_df, preset).astype(bool)
    reference_pred = pd.Series(output["Churn"].astype("float64").values, index=output.index)
    calibrated_pred = pd.Series(np.nan, index=output.index, dtype="float64")
    calibrated_pred.loc[specialist_mask] = _predict_local_calibrator(
        calibrator,
        reference_pred.loc[specialist_mask],
    ).values

    output["specialist_mask"] = specialist_mask.astype("int8").values
    output["calibrated_pred"] = calibrated_pred.values
    output.loc[specialist_mask, "Churn"] = (
        (1.0 - alpha) * output.loc[specialist_mask, "Churn"].astype("float64")
        + alpha * calibrated_pred.loc[specialist_mask]
    )
    return output
