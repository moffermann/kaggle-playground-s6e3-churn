"""Helpers to compare challenger residual chains directly against incumbent v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN
from .diagnostics import OOF_TARGET_COLUMN
from .evaluation import binary_auc
from .validation_protocol import DOMINANT_MACROFAMILY, SUPPORTED_STAGES, evaluate_validation_protocol


V3_ORDER: tuple[str, ...] = (
    "early_all_internet",
    "fiber_paperless_early",
    "late_mtm_fiber",
    "late_mtm_fiber_paperless",
)

V3_STEP_OOF_PATHS: dict[str, str] = {
    "early_all_internet": "artifacts/reports/residual_reranker_early_all_internet_midcap_oof.csv",
    "fiber_paperless_early": "artifacts/reports/residual_reranker_fiber_paperless_early_teacher_midcap_oof.csv",
    "late_mtm_fiber": "artifacts/reports/residual_reranker_late_mtm_fiber_teacher_midcap_oof.csv",
    "late_mtm_fiber_paperless": "artifacts/reports/residual_reranker_late_mtm_fiber_paperless_teacher_midcap_oof.csv",
}

_STEP_REQUIRED_COLUMNS = {
    ID_COLUMN,
    OOF_TARGET_COLUMN,
    "specialist_mask",
    "candidate_pred",
    "reference_pred",
}
_REFERENCE_ATOL = 1e-8


def _load_step_oof_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(_STEP_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} must contain columns {sorted(_STEP_REQUIRED_COLUMNS)}; missing {missing}")
    return frame[list(_STEP_REQUIRED_COLUMNS)].sort_values(ID_COLUMN).reset_index(drop=True)


def _validate_specialist_mask(frame: pd.DataFrame, *, step_name: str) -> None:
    mask = frame["specialist_mask"]
    if mask.isna().any():
        raise ValueError(f"specialist_mask contains nulls in step '{step_name}'")
    normalized = pd.to_numeric(mask, errors="coerce")
    if normalized.isna().any():
        invalid = sorted({str(value) for value in mask[normalized.isna()].unique()})
        raise ValueError(f"specialist_mask must be boolean/0/1 in step '{step_name}'; invalid values: {invalid}")
    if not np.all(np.isclose(normalized.to_numpy(dtype="float64"), np.round(normalized.to_numpy(dtype="float64")), atol=0.0)):
        invalid = sorted({str(value) for value in mask[~np.isclose(normalized, np.round(normalized), atol=0.0)].unique()})
        raise ValueError(
            f"specialist_mask must be boolean/0/1 in step '{step_name}'; non-integer values: {invalid}"
        )
    valid_values = set(np.round(normalized.to_numpy(dtype="float64")).astype(int).tolist())
    if not valid_values.issubset({0, 1}):
        raise ValueError(
            f"specialist_mask must contain only 0/1 in step '{step_name}'; got {sorted(valid_values)}"
        )


def _merge_step_oof_paths(candidate_step_oof_paths: Mapping[str, str | Path] | None) -> dict[str, str]:
    merged = {name: str(path) for name, path in V3_STEP_OOF_PATHS.items()}
    if candidate_step_oof_paths:
        merged.update({str(name): str(path) for name, path in candidate_step_oof_paths.items()})
    return merged


def load_chain_step_frames(
    candidate_step_oof_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the default v3 step OOF frames plus any challenger overrides."""
    step_paths = _merge_step_oof_paths(candidate_step_oof_paths)
    loaded = {name: _load_step_oof_frame(path) for name, path in step_paths.items()}
    _assert_compatible_step_frames(loaded)
    return loaded


def _assert_compatible_step_frames(step_frames: Mapping[str, pd.DataFrame]) -> None:
    base_ids = None
    base_target = None
    base_reference = None

    for name, frame in step_frames.items():
        _validate_specialist_mask(frame, step_name=name)
        ids = frame[ID_COLUMN].to_numpy()
        target = frame[OOF_TARGET_COLUMN].astype(int).to_numpy()
        reference = frame["reference_pred"].astype(float).to_numpy()
        if base_ids is None:
            base_ids = ids
            base_target = target
            base_reference = reference
            continue
        if not np.array_equal(base_ids, ids):
            raise ValueError(f"id mismatch across step OOF frames; first mismatch at '{name}'")
        if not np.array_equal(base_target, target):
            raise ValueError(f"target mismatch across step OOF frames; first mismatch at '{name}'")
        if not np.allclose(base_reference, reference, atol=_REFERENCE_ATOL, rtol=0.0):
            max_abs_diff = float(np.max(np.abs(base_reference - reference)))
            raise ValueError(
                f"reference_pred mismatch across step OOF frames; first mismatch at '{name}', "
                f"max_abs_diff={max_abs_diff:.3e}, atol={_REFERENCE_ATOL:.1e}"
            )


def _assert_same_incumbent_reference(
    *,
    reference_frames: Mapping[str, pd.DataFrame],
    candidate_frames: Mapping[str, pd.DataFrame],
) -> None:
    reference_base = _get_base_reference(reference_frames).to_numpy(dtype="float64")
    candidate_base = _get_base_reference(candidate_frames).to_numpy(dtype="float64")
    if not np.allclose(reference_base, candidate_base, atol=_REFERENCE_ATOL, rtol=0.0):
        max_abs_diff = float(np.max(np.abs(reference_base - candidate_base)))
        raise ValueError(
            "candidate step overrides changed the incumbent reference_pred base; "
            f"max_abs_diff={max_abs_diff:.3e}, atol={_REFERENCE_ATOL:.1e}"
        )


def _validate_full_train_coverage(ids: pd.Series, *, train_csv_path: str | Path) -> None:
    train_ids = (
        pd.read_csv(train_csv_path, usecols=[ID_COLUMN])[ID_COLUMN]
        .sort_values()
        .reset_index(drop=True)
        .astype(ids.dtype)
    )
    candidate_ids = ids.sort_values().reset_index(drop=True)
    if len(candidate_ids) != len(train_ids):
        raise ValueError(
            "step OOF coverage does not match train.csv row count: "
            f"{len(candidate_ids)} vs {len(train_ids)}"
        )
    if not np.array_equal(candidate_ids.to_numpy(), train_ids.to_numpy()):
        raise ValueError("step OOF ids do not exactly cover train.csv ids")


def _get_base_reference(step_frames: Mapping[str, pd.DataFrame]) -> pd.Series:
    any_frame = next(iter(step_frames.values()))
    return pd.Series(
        any_frame["reference_pred"].astype(float).to_numpy(),
        index=any_frame[ID_COLUMN].to_numpy(),
        dtype="float64",
        name="reference_pred",
    )


def _get_target_series(step_frames: Mapping[str, pd.DataFrame]) -> pd.Series:
    any_frame = next(iter(step_frames.values()))
    return pd.Series(
        any_frame[OOF_TARGET_COLUMN].astype(int).to_numpy(),
        index=any_frame[ID_COLUMN].to_numpy(),
        dtype="int8",
        name=OOF_TARGET_COLUMN,
    )


def build_chain_prediction(
    order: Sequence[str],
    step_frames: Mapping[str, pd.DataFrame],
) -> pd.Series:
    """Apply a challenger chain over the shared base reference prediction."""
    if not order:
        raise ValueError("order must contain at least one step name")

    reference = _get_base_reference(step_frames)
    current = reference.copy()

    for name in order:
        if name not in step_frames:
            available = sorted(step_frames)
            raise ValueError(f"Unknown step '{name}'. Available steps: {available}")
        frame = step_frames[name]
        mask = frame["specialist_mask"].astype(bool).to_numpy()
        candidate = frame["candidate_pred"].astype(float).to_numpy()
        current.loc[mask] = candidate[mask]
    return current


def compute_repeated_cv_auc_stats(
    target: pd.Series,
    prediction: pd.Series,
    *,
    repeats: int = 5,
    folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute repeated-fold AUC summary for a fixed OOF prediction vector."""
    y = target.astype(int).to_numpy()
    pred = prediction.astype(float).to_numpy()
    fold_aucs: list[float] = []

    for repeat in range(int(repeats)):
        splitter = StratifiedKFold(
            n_splits=int(folds),
            shuffle=True,
            random_state=int(random_state) + repeat,
        )
        for _, valid_idx in splitter.split(np.zeros_like(y), y):
            fold_aucs.append(float(binary_auc(y[valid_idx], pred[valid_idx])))

    metrics = np.asarray(fold_aucs, dtype="float64")
    return {
        "repeats": int(repeats),
        "folds": int(folds),
        "random_state": int(random_state),
        "cv_mean_auc": float(metrics.mean()),
        "cv_std_auc": float(metrics.std(ddof=0)),
        "cv_min_auc": float(metrics.min()),
        "cv_max_auc": float(metrics.max()),
        "full_auc": float(binary_auc(y, pred)),
        "fold_aucs": [float(value) for value in metrics.tolist()],
    }


def evaluate_candidate_chain_against_v3(
    *,
    candidate_order: Sequence[str],
    candidate_step_oof_paths: Mapping[str, str | Path] | None = None,
    stage: str = "smoke",
    train_csv_path: str | Path = "data/raw/train.csv",
    test_csv_path: str | Path = "data/raw/test.csv",
    target_family_level: str = "segment3",
    target_family_value: str = DOMINANT_MACROFAMILY,
    dominant_family_value: str = DOMINANT_MACROFAMILY,
    label: str = "candidate",
    out_dir: str | Path = "artifacts/reports",
) -> dict[str, Any]:
    """Compare a challenger residual chain directly against incumbent v3."""
    if str(stage).strip().lower() not in SUPPORTED_STAGES:
        raise ValueError(f"Unsupported stage '{stage}'. Supported: {SUPPORTED_STAGES}")
    if not candidate_order:
        raise ValueError("candidate_order must contain at least one step name")

    out_root = Path(out_dir)
    reference_step_frames = load_chain_step_frames()
    candidate_step_frames = load_chain_step_frames(candidate_step_oof_paths)
    _assert_same_incumbent_reference(
        reference_frames=reference_step_frames,
        candidate_frames=candidate_step_frames,
    )
    target = _get_target_series(reference_step_frames)
    ids = pd.Series(target.index.to_numpy(), index=target.index, dtype="int64", name=ID_COLUMN)
    _validate_full_train_coverage(ids, train_csv_path=train_csv_path)

    v3_pred = build_chain_prediction(V3_ORDER, reference_step_frames)
    candidate_pred = build_chain_prediction(candidate_order, candidate_step_frames)
    delta_auc = float(binary_auc(target, candidate_pred) - binary_auc(target, v3_pred))

    v3_frame = pd.DataFrame(
        {
            ID_COLUMN: ids.values,
            OOF_TARGET_COLUMN: target.values,
            "candidate_pred": v3_pred.values,
        }
    )
    candidate_frame = pd.DataFrame(
        {
            ID_COLUMN: ids.values,
            OOF_TARGET_COLUMN: target.values,
            "candidate_pred": candidate_pred.values,
        }
    )

    analysis_frame = pd.DataFrame(
        {
            ID_COLUMN: ids.values,
            OOF_TARGET_COLUMN: target.values,
            "reference_pred": v3_pred.values,
            "candidate_pred": candidate_pred.values,
        }
    )

    v3_oof_path = ensure_parent_dir(out_root / "validation_protocol_v3_chain_oof.csv")
    candidate_oof_path = ensure_parent_dir(out_root / f"validation_protocol_{label}_chain_oof.csv")
    analysis_oof_path = ensure_parent_dir(out_root / f"validation_protocol_{label}_vs_v3_analysis_oof.csv")
    v3_frame.to_csv(v3_oof_path, index=False)
    candidate_frame.to_csv(candidate_oof_path, index=False)
    analysis_frame.to_csv(analysis_oof_path, index=False)

    reference_metrics = {
        "generated_for": "validation_protocol_v3_chain_reference",
        "order": list(V3_ORDER),
        **compute_repeated_cv_auc_stats(target, v3_pred),
    }
    candidate_metrics = {
        "generated_for": "validation_protocol_chain_candidate",
        "order": [str(step) for step in candidate_order],
        **compute_repeated_cv_auc_stats(target, candidate_pred),
    }
    reference_metrics_path = ensure_parent_dir(out_root / "validation_protocol_v3_chain_metrics.json")
    candidate_metrics_path = ensure_parent_dir(out_root / f"validation_protocol_{label}_chain_metrics.json")
    write_json(reference_metrics_path, reference_metrics)
    write_json(candidate_metrics_path, candidate_metrics)

    verdict_path = ensure_parent_dir(out_root / f"validation_protocol_{label}_vs_v3_{stage}.json")
    protocol_result = evaluate_validation_protocol(
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        stage=stage,
        analysis_oof_path=analysis_oof_path,
        target_family_level=target_family_level,
        target_family_value=target_family_value,
        dominant_family_value=dominant_family_value,
        candidate_metrics_json=candidate_metrics_path,
        reference_metrics_json=reference_metrics_path,
        out_json_path=verdict_path,
    )

    summary = {
        "label": str(label),
        "stage": str(stage),
        "candidate_order": [str(step) for step in candidate_order],
        "v3_order": list(V3_ORDER),
        "delta_vs_v3_oof_auc": delta_auc,
        "v3_oof_path": str(v3_oof_path),
        "candidate_oof_path": str(candidate_oof_path),
        "analysis_oof_path": str(analysis_oof_path),
        "reference_metrics_path": str(reference_metrics_path),
        "candidate_metrics_path": str(candidate_metrics_path),
        "verdict_path": str(verdict_path),
        "verdict": protocol_result["verdict"],
        "overall_metrics": protocol_result["overall_metrics"],
    }
    summary_path = ensure_parent_dir(out_root / f"validation_protocol_{label}_vs_v3_summary.json")
    write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
