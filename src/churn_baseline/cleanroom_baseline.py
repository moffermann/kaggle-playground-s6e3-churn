"""Clean-room baseline smoke suite compared directly against incumbent v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import ensure_parent_dir, write_json
from .config import CatBoostHyperParams, ID_COLUMN
from .diagnostics import OOF_TARGET_COLUMN, load_reference_prediction_frame, utc_now_iso
from .evaluation import binary_auc
from .feature_engineering import normalize_feature_blocks
from .incumbent_v3 import compute_repeated_cv_auc_stats
from .noise_audit import DEFAULT_V3_OOF_SPEC, DOMINANT_MACROFAMILY
from .pipeline import train_baseline_cv
from .validation_protocol import evaluate_validation_protocol


@dataclass(frozen=True)
class CleanroomConfig:
    label: str
    feature_blocks: tuple[str, ...]
    stratify_mode: str = "target"


DEFAULT_CLEANROOM_CONFIGS: tuple[CleanroomConfig, ...] = (
    CleanroomConfig(label="cb_raw", feature_blocks=()),
    CleanroomConfig(label="cb_r", feature_blocks=("R",)),
    CleanroomConfig(label="cb_rv", feature_blocks=("R", "V")),
)


def _scan_simple_blend(reference_pred: pd.Series, candidate_pred: pd.Series, target: pd.Series) -> dict[str, Any]:
    best = {
        "best_alpha": 0.0,
        "best_auc": float(binary_auc(target, reference_pred)),
        "delta_vs_reference": 0.0,
    }
    for alpha in (0.05, 0.10, 0.15, 0.20, 0.25):
        blended = ((1.0 - alpha) * reference_pred.astype(float)) + (alpha * candidate_pred.astype(float))
        auc = float(binary_auc(target, blended))
        delta = float(auc - best["best_auc"] + best["delta_vs_reference"])
        if auc > best["best_auc"] + 1e-12:
            best = {
                "best_alpha": float(alpha),
                "best_auc": auc,
                "delta_vs_reference": float(auc - float(binary_auc(target, reference_pred))),
            }
    return best


def run_cleanroom_baseline_suite(
    *,
    train_csv_path: str | Path,
    v3_oof_spec: str = DEFAULT_V3_OOF_SPEC,
    out_dir: str | Path = "artifacts/reports",
    model_dir: str | Path = "artifacts/models",
    configs: Sequence[CleanroomConfig] = DEFAULT_CLEANROOM_CONFIGS,
    params: CatBoostHyperParams | None = None,
    folds: int = 2,
    random_state: int = 42,
    early_stopping_rounds: int = 60,
    verbose: int = 0,
) -> dict[str, Any]:
    if params is None:
        params = CatBoostHyperParams(
            iterations=150,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=random_state,
        )

    out_root = Path(out_dir)
    model_root = Path(model_dir)
    reference_frame, _ = load_reference_prediction_frame(
        reference_oof_spec=v3_oof_spec,
        id_column=ID_COLUMN,
        target_column=OOF_TARGET_COLUMN,
    )
    reference_target = reference_frame[OOF_TARGET_COLUMN].astype(int)
    reference_pred = reference_frame["reference_pred"].astype(float)
    reference_metrics = compute_repeated_cv_auc_stats(reference_target, reference_pred)
    reference_metrics_path = ensure_parent_dir(out_root / "cleanroom_reference_v3_metrics.json")
    write_json(reference_metrics_path, reference_metrics)

    rows: list[dict[str, Any]] = []
    for config in configs:
        label = f"cleanroom_{config.label}_smoke"
        model_path = model_root / f"{label}.cbm"
        metrics_path = out_root / f"{label}_metrics.json"
        oof_path = out_root / f"{label}_oof.csv"
        analysis_path = out_root / f"{label}_analysis_oof.csv"
        candidate_metrics_path = out_root / f"{label}_candidate_metrics.json"
        verdict_path = out_root / f"validation_protocol_{label}_vs_v3_smoke.json"

        metrics = train_baseline_cv(
            train_csv_path=train_csv_path,
            model_path=model_path,
            metrics_path=metrics_path,
            oof_path=oof_path,
            params=params,
            folds=folds,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            feature_blocks=config.feature_blocks,
            stratify_mode=config.stratify_mode,
        )

        candidate_frame = pd.read_csv(oof_path)
        analysis_frame = reference_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "reference_pred"]].merge(
            candidate_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "oof_pred"]].rename(columns={"oof_pred": "candidate_pred"}),
            how="inner",
            on=[ID_COLUMN, OOF_TARGET_COLUMN],
            validate="one_to_one",
        )
        analysis_frame.to_csv(analysis_path, index=False)

        candidate_metrics = compute_repeated_cv_auc_stats(
            analysis_frame[OOF_TARGET_COLUMN],
            analysis_frame["candidate_pred"],
        )
        write_json(candidate_metrics_path, candidate_metrics)
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
        blend_scan = _scan_simple_blend(
            reference_pred=analysis_frame["reference_pred"],
            candidate_pred=analysis_frame["candidate_pred"],
            target=analysis_frame[OOF_TARGET_COLUMN].astype(int),
        )
        rows.append(
            {
                "label": label,
                "feature_blocks": list(normalize_feature_blocks(config.feature_blocks)),
                "stratify_mode": config.stratify_mode,
                "oof_auc": float(metrics["oof_auc"]),
                "delta_vs_v3_oof_auc": float(candidate_metrics["full_auc"] - reference_metrics["full_auc"]),
                "validation_verdict": verdict["verdict"],
                "best_blend_alpha_vs_v3": float(blend_scan["best_alpha"]),
                "best_blend_delta_vs_v3": float(blend_scan["delta_vs_reference"]),
                "metrics_path": str(metrics_path),
                "oof_path": str(oof_path),
                "analysis_oof_path": str(analysis_path),
                "validation_verdict_path": str(verdict_path),
            }
        )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "train_csv_path": str(train_csv_path),
        "reference_v3_oof_spec": str(v3_oof_spec),
        "reference_oof_auc": float(reference_metrics["full_auc"]),
        "configs": rows,
        "best_candidate": max(rows, key=lambda item: item["delta_vs_v3_oof_auc"]) if rows else None,
        "best_blend_candidate": max(rows, key=lambda item: item["best_blend_delta_vs_v3"]) if rows else None,
    }
    return summary
