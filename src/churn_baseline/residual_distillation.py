"""Residual-family distillation experiments against incumbent v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold

from .artifacts import ensure_parent_dir, write_json
from .config import CatBoostHyperParams, ID_COLUMN, TARGET_COLUMN
from .data import infer_categorical_columns, load_csv
from .diagnostics import OOF_TARGET_COLUMN
from .evaluation import binary_auc
from .incumbent_v3 import (
    V3_ORDER,
    _get_base_reference,
    _get_target_series,
    build_chain_prediction,
    compute_repeated_cv_auc_stats,
    load_chain_step_frames,
)
from .feature_engineering import normalize_feature_blocks, partition_feature_blocks
from .noise_audit import DOMINANT_MACROFAMILY
from .pipeline import (
    _prepare_test_matrix,
    _prepare_train_matrix,
    _transform_pair_with_stateful_blocks,
    _transform_single_with_stateful_blocks,
)
from .specialist import _append_reference_features, _clip_probability, _load_reference_submission_frame
from .validation_protocol import evaluate_validation_protocol

DISTILLATION_BASE_REFERENCE_SUBMISSION_PATH = "artifacts/submissions/playground-series-s6e3-rvblend.csv"


@dataclass(frozen=True)
class ResidualDistillationConfig:
    """Configuration for the minimal total residual distillation smoke."""

    label: str = "residual_distillation_smoke"
    feature_blocks: tuple[str, ...] = ("H", "R", "S", "V")
    alpha_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def _align_series_by_train_ids(
    train_df: pd.DataFrame,
    values_by_id: pd.Series,
    *,
    name: str,
) -> pd.Series:
    aligned = train_df[ID_COLUMN].map(values_by_id)
    if aligned.isna().any():
        missing_ids = train_df.loc[aligned.isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"{name} missing ids from train: {missing_ids}")
    return pd.Series(aligned.astype("float64").values, index=train_df.index, dtype="float64", name=name)


def _build_reference_frames(
    train_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    step_frames = load_chain_step_frames()
    base_reference_by_id = _get_base_reference(step_frames)
    target_by_id = _get_target_series(step_frames).astype("int8")
    v3_pred_by_id = build_chain_prediction(V3_ORDER, step_frames)

    base_reference = _align_series_by_train_ids(train_df, base_reference_by_id, name="base_reference_pred")
    v3_pred = _align_series_by_train_ids(train_df, v3_pred_by_id, name="v3_pred")

    target = train_df[ID_COLUMN].map(target_by_id)
    if target.isna().any():
        missing_ids = train_df.loc[target.isna(), ID_COLUMN].head(5).tolist()
        raise ValueError(f"incumbent target missing ids from train: {missing_ids}")
    return (
        pd.Series(target.astype("int8").values, index=train_df.index, dtype="int8", name=OOF_TARGET_COLUMN),
        base_reference,
        v3_pred,
    )


def _scan_alpha_against_v3(
    *,
    y: pd.Series,
    base_reference: pd.Series,
    distilled_delta_pred: pd.Series,
    alpha_grid: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.Series]:
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one value")

    scan_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_candidate: pd.Series | None = None
    for alpha in alpha_grid:
        alpha_value = float(alpha)
        candidate = _clip_probability(base_reference + alpha_value * distilled_delta_pred)
        auc = float(binary_auc(y, candidate))
        row = {"alpha": alpha_value, "candidate_oof_auc": auc}
        scan_rows.append(row)
        if best_row is None or auc > float(best_row["candidate_oof_auc"]) + 1e-12:
            best_row = row
            best_candidate = candidate

    assert best_row is not None
    assert best_candidate is not None
    return scan_rows, best_row, best_candidate


def run_total_residual_distillation_smoke(
    *,
    train_csv_path: str | Path,
    out_dir: str | Path = "artifacts/reports",
    model_dir: str | Path = "artifacts/models",
    config: ResidualDistillationConfig = ResidualDistillationConfig(),
    params: CatBoostHyperParams | None = None,
    folds: int = 2,
    random_state: int = 42,
    early_stopping_rounds: int = 60,
    verbose: int = 0,
) -> dict[str, Any]:
    """Distill the total v3 residual over the base reference and gate it against v3."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if params is None:
        params = CatBoostHyperParams(
            iterations=150,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=random_state,
        )

    label = str(config.label)
    out_root = Path(out_dir)
    model_root = Path(model_dir)
    train_df = load_csv(train_csv_path)
    target, base_reference, v3_pred = _build_reference_frames(train_df)
    distill_target = pd.Series(
        v3_pred.to_numpy(dtype="float64") - base_reference.to_numpy(dtype="float64"),
        index=train_df.index,
        dtype="float64",
        name="distill_target",
    )

    _, _, stateful_blocks, x, y = _prepare_train_matrix(train_df, config.feature_blocks)
    if not np.array_equal(y.astype("int8").values, target.astype("int8").values):
        raise ValueError("Competition target and incumbent target are misaligned.")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    distilled_delta_pred = pd.Series(index=x.index, dtype="float64", name="distilled_delta_pred")
    fold_rows: list[dict[str, Any]] = []
    fold_iterations: list[int] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        x_valid = x.iloc[valid_idx]
        x_train, x_valid = _transform_pair_with_stateful_blocks(x_train, x_valid, stateful_blocks)

        train_reference = base_reference.iloc[train_idx]
        valid_reference = base_reference.iloc[valid_idx]
        x_train, _ = _append_reference_features(
            x_train,
            reference_pred=train_reference,
            reference_component_frame=None,
            include_logit=True,
        )
        x_valid, _ = _append_reference_features(
            x_valid,
            reference_pred=valid_reference,
            reference_component_frame=None,
            include_logit=True,
        )

        cat_columns = infer_categorical_columns(x_train)
        fold_model = CatBoostRegressor(
            iterations=params.iterations,
            learning_rate=params.learning_rate,
            depth=params.depth,
            l2_leaf_reg=params.l2_leaf_reg,
            random_seed=int(random_state) + fold_number,
            loss_function="RMSE",
            eval_metric="RMSE",
            verbose=verbose,
        )
        fold_model.fit(
            x_train,
            distill_target.iloc[train_idx],
            cat_features=cat_columns,
            eval_set=(x_valid, distill_target.iloc[valid_idx]),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        valid_delta_pred = pd.Series(
            fold_model.predict(x_valid),
            index=x_valid.index,
            dtype="float64",
        )
        distilled_delta_pred.iloc[valid_idx] = valid_delta_pred.values
        final_iterations = max(int(fold_model.get_best_iteration()) + 1, 1)
        fold_iterations.append(final_iterations)

        candidate_valid = _clip_probability(valid_reference + valid_delta_pred)
        v3_valid = v3_pred.iloc[valid_idx]
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "v3_auc": float(binary_auc(y_valid, v3_valid)),
                "candidate_auc": float(binary_auc(y_valid, candidate_valid)),
                "delta_vs_v3_auc": float(binary_auc(y_valid, candidate_valid) - binary_auc(y_valid, v3_valid)),
                "delta_target_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                valid_delta_pred.to_numpy(dtype="float64")
                                - distill_target.iloc[valid_idx].to_numpy(dtype="float64")
                            )
                            ** 2
                        )
                    )
                ),
                "final_iterations": int(final_iterations),
            }
        )

    if distilled_delta_pred.isna().any():
        raise RuntimeError("Distilled OOF delta contains missing values.")

    alpha_scan_rows, best_alpha_row, candidate_best = _scan_alpha_against_v3(
        y=y,
        base_reference=base_reference,
        distilled_delta_pred=distilled_delta_pred,
        alpha_grid=config.alpha_grid,
    )

    x_full = _transform_single_with_stateful_blocks(x, stateful_blocks)
    x_full, _ = _append_reference_features(
        x_full,
        reference_pred=base_reference,
        reference_component_frame=None,
        include_logit=True,
    )
    full_iterations = max(int(round(float(np.mean(fold_iterations)))), 1)
    cat_columns = infer_categorical_columns(x_full)
    full_model = CatBoostRegressor(
        iterations=full_iterations,
        learning_rate=params.learning_rate,
        depth=params.depth,
        l2_leaf_reg=params.l2_leaf_reg,
        random_seed=params.random_seed,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=verbose,
    )
    full_model.fit(
        x_full,
        distill_target,
        cat_features=cat_columns,
        verbose=verbose,
    )
    model_path = ensure_parent_dir(model_root / f"{label}.cbm")
    full_model.save_model(str(model_path))

    analysis_frame = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            OOF_TARGET_COLUMN: y.astype("int8").values,
            "reference_pred": v3_pred.to_numpy(dtype="float64"),
            "candidate_pred": candidate_best.to_numpy(dtype="float64"),
            "base_reference_pred": base_reference.to_numpy(dtype="float64"),
            "distill_target": distill_target.to_numpy(dtype="float64"),
            "distilled_delta_pred": distilled_delta_pred.to_numpy(dtype="float64"),
        }
    )
    analysis_path = ensure_parent_dir(out_root / f"{label}_analysis_oof.csv")
    analysis_frame.to_csv(analysis_path, index=False)

    reference_metrics = compute_repeated_cv_auc_stats(y, v3_pred)
    candidate_metrics = compute_repeated_cv_auc_stats(y, candidate_best)
    reference_metrics_path = ensure_parent_dir(out_root / f"{label}_reference_v3_metrics.json")
    candidate_metrics_path = ensure_parent_dir(out_root / f"{label}_candidate_metrics.json")
    write_json(reference_metrics_path, reference_metrics)
    write_json(candidate_metrics_path, candidate_metrics)

    verdict_path = ensure_parent_dir(out_root / f"validation_protocol_{label}_vs_v3_smoke.json")
    verdict = evaluate_validation_protocol(
        train_csv_path=train_csv_path,
        test_csv_path="data/raw/test.csv",
        stage="smoke",
        analysis_oof_path=analysis_path,
        target_family_level="segment3",
        target_family_value=DOMINANT_MACROFAMILY,
        dominant_family_value=DOMINANT_MACROFAMILY,
        candidate_metrics_json=candidate_metrics_path,
        reference_metrics_json=reference_metrics_path,
        out_json_path=verdict_path,
    )

    summary = {
        "label": label,
        "train_csv_path": str(train_csv_path),
        "expected_reference_submission_path": DISTILLATION_BASE_REFERENCE_SUBMISSION_PATH,
        "feature_blocks": list(config.feature_blocks),
        "alpha_grid": [float(alpha) for alpha in config.alpha_grid],
        "alpha_scan": alpha_scan_rows,
        "best_alpha": float(best_alpha_row["alpha"]),
        "candidate_oof_auc": float(candidate_metrics["full_auc"]),
        "reference_oof_auc": float(reference_metrics["full_auc"]),
        "delta_vs_v3_oof_auc": float(candidate_metrics["full_auc"] - reference_metrics["full_auc"]),
        "distill_target_mean_abs": float(np.mean(np.abs(distill_target.to_numpy(dtype="float64")))),
        "distilled_delta_mean_abs": float(np.mean(np.abs(distilled_delta_pred.to_numpy(dtype="float64")))),
        "fold_metrics": fold_rows,
        "fold_final_iterations": [int(value) for value in fold_iterations],
        "full_iterations": int(full_iterations),
        "model_path": str(model_path),
        "analysis_oof_path": str(analysis_path),
        "reference_metrics_path": str(reference_metrics_path),
        "candidate_metrics_path": str(candidate_metrics_path),
        "validation_verdict_path": str(verdict_path),
        "validation_verdict": verdict["verdict"],
    }
    metrics_path = ensure_parent_dir(out_root / f"{label}_metrics.json")
    write_json(metrics_path, summary)
    summary["metrics_path"] = str(metrics_path)
    return summary


def make_residual_distillation_submission(
    *,
    test_csv_path: str | Path,
    train_csv_path: str | Path,
    reference_submission_path: str | Path,
    expected_reference_submission_path: str | Path | None,
    model_path: str | Path,
    feature_blocks: Sequence[str] | None,
    alpha: float,
    output_csv_path: str | Path,
    report_json_path: str | Path,
) -> dict[str, Any]:
    """Materialize a submission by adding a distilled residual over a base teacher submission."""
    normalized_blocks = normalize_feature_blocks(feature_blocks or ())
    stateless_blocks, stateful_blocks = partition_feature_blocks(normalized_blocks)
    alpha_value = float(alpha)
    if expected_reference_submission_path is not None:
        supplied_reference = Path(reference_submission_path).resolve()
        expected_reference = Path(expected_reference_submission_path).resolve()
        if supplied_reference != expected_reference:
            raise ValueError(
                "Residual distillation submit path requires the same base reference lineage used during training: "
                f"expected '{expected_reference}', got '{supplied_reference}'."
            )

    test_df = load_csv(test_csv_path)
    reference_frame = _load_reference_submission_frame(
        test_df=test_df,
        reference_submission_path=reference_submission_path,
    )
    reference_target_column = TARGET_COLUMN if TARGET_COLUMN in reference_frame.columns else "Churn"
    if reference_target_column not in reference_frame.columns:
        raise ValueError(
            f"{reference_submission_path} must contain a prediction column compatible with '{TARGET_COLUMN}'."
        )
    x_test = _prepare_test_matrix(
        test_df,
        stateless_blocks=stateless_blocks,
        stateful_blocks=stateful_blocks,
        train_csv_path=train_csv_path,
    )
    base_reference = pd.Series(
        reference_frame[reference_target_column].astype("float64").values,
        index=x_test.index,
        dtype="float64",
        name="base_reference_pred",
    )
    x_test, _ = _append_reference_features(
        x_test,
        reference_pred=base_reference,
        reference_component_frame=None,
        include_logit=True,
    )

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    expected_columns = list(getattr(model, "feature_names_", []))
    if expected_columns:
        missing = [column for column in expected_columns if column not in x_test.columns]
        if missing:
            raise ValueError(f"Missing inference columns for residual distillation model: {missing}")
        x_test = x_test.loc[:, expected_columns]

    distilled_delta = pd.Series(
        model.predict(x_test),
        index=x_test.index,
        dtype="float64",
        name="distilled_delta_pred",
    )
    candidate_pred = _clip_probability(base_reference + alpha_value * distilled_delta)

    submission = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN].values,
            TARGET_COLUMN: candidate_pred.to_numpy(dtype="float64"),
        }
    )
    output_path = ensure_parent_dir(output_csv_path)
    submission.to_csv(output_path, index=False)

    delta = candidate_pred - base_reference
    report = {
        "test_csv_path": str(test_csv_path),
        "train_csv_path": str(train_csv_path),
        "reference_submission_path": str(reference_submission_path),
        "expected_reference_submission_path": (
            str(expected_reference_submission_path) if expected_reference_submission_path is not None else None
        ),
        "model_path": str(model_path),
        "feature_blocks": list(normalized_blocks),
        "alpha": alpha_value,
        "rows": int(len(submission)),
        "mean_abs_shift_vs_base": float(delta.abs().mean()),
        "max_abs_shift_vs_base": float(delta.abs().max()),
        "mean_signed_shift_vs_base": float(delta.mean()),
        "base_prediction_min": float(base_reference.min()),
        "base_prediction_max": float(base_reference.max()),
        "candidate_prediction_min": float(candidate_pred.min()),
        "candidate_prediction_max": float(candidate_pred.max()),
        "delta_prediction_min": float(distilled_delta.min()),
        "delta_prediction_max": float(distilled_delta.max()),
        "output_csv_path": str(output_path),
    }
    write_json(report_json_path, report)
    return report
