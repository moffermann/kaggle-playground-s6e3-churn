"""Aggressive diagnostics for why incumbent v3 dominates challengers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN
from .data import load_csv
from .diagnostics import (
    OOF_TARGET_COLUMN,
    build_family_frame,
    describe_file,
    load_reference_prediction_frame,
    utc_now_iso,
)
from .evaluation import binary_auc
from .validation_protocol import DOMINANT_MACROFAMILY


DEFAULT_V3_OOF_SPEC = "artifacts/reports/validation_protocol_v3_chain_oof.csv#candidate_pred"


@dataclass(frozen=True)
class ChallengerSpec:
    """Descriptor for one challenger OOF artifact."""

    name: str
    path: str
    prediction_column: str = "oof_pred"
    family: str = "unknown"


DEFAULT_CHALLENGERS: tuple[ChallengerSpec, ...] = (
    ChallengerSpec(
        name="global_feature_block_rv_issue7",
        path="artifacts/reports/fe_blockRV_issue7_smoke_oof.csv",
        family="global_feature_block",
    ),
    ChallengerSpec(
        name="family_gated_ec_mtm_fiber",
        path="artifacts/reports/family_gated_ec_mtm_fiber_any_teacher_smoke_oof.csv",
        prediction_column="candidate_pred",
        family="family_challenger",
    ),
    ChallengerSpec(
        name="near_duplicate_drop",
        path="artifacts/reports/noise_mitigation_drop_smoke_oof.csv",
        family="data_centric",
    ),
    ChallengerSpec(
        name="external_telco_transfer",
        path="artifacts/reports/telco_transfer_smoke_oof.csv",
        family="external_transfer",
    ),
    ChallengerSpec(
        name="residual_ec_mtm_fiber_paperless_any",
        path="artifacts/reports/residual_reranker_ec_mtm_fiber_paperless_any_teacher_smoke_oof.csv",
        prediction_column="candidate_pred",
        family="near_pass_residual",
    ),
)


def _clip_probability(values: pd.Series, epsilon: float = 1e-6) -> pd.Series:
    return values.astype("float64").clip(epsilon, 1.0 - epsilon)


def _safe_binary_auc(y_true: pd.Series, pred: pd.Series) -> float:
    if int(y_true.nunique(dropna=False)) < 2:
        return float("nan")
    return float(binary_auc(y_true, pred))


def _logloss_vector(y_true: pd.Series, pred: pd.Series) -> pd.Series:
    clipped = _clip_probability(pred)
    target = y_true.astype("float64")
    return -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))


def _load_candidate_frame(spec: ChallengerSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.path)
    required = {ID_COLUMN, OOF_TARGET_COLUMN, spec.prediction_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{spec.path} missing required columns {missing}")
    out = frame[[ID_COLUMN, OOF_TARGET_COLUMN, spec.prediction_column]].copy()
    out = out.rename(columns={spec.prediction_column: "candidate_pred"})
    return out


def _validate_pair(reference_frame: pd.DataFrame, candidate_frame: pd.DataFrame, *, challenger_name: str) -> pd.DataFrame:
    merged = reference_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "reference_pred"]].merge(
        candidate_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "candidate_pred"]],
        how="inner",
        on=[ID_COLUMN, OOF_TARGET_COLUMN],
        validate="one_to_one",
    )
    if len(merged) != len(reference_frame):
        raise ValueError(
            f"Candidate '{challenger_name}' does not fully cover the v3 reference rows: "
            f"{len(merged)} vs {len(reference_frame)}"
        )
    return merged.sort_values(ID_COLUMN).reset_index(drop=True)


def _bucket_from_edges(values: pd.Series, edges: Sequence[float], *, prefix: str) -> pd.Series:
    numeric_edges = np.unique(np.asarray(list(edges), dtype="float64"))
    if len(numeric_edges) < 2:
        numeric_edges = np.asarray([0.0, 1.0], dtype="float64")
    labels = [f"{prefix}_{idx:02d}" for idx in range(len(numeric_edges) - 1)]
    bucket = pd.cut(
        values.astype("float64"),
        bins=numeric_edges,
        include_lowest=True,
        labels=labels,
        duplicates="drop",
    )
    return bucket.astype(str).replace("nan", "__missing__")


def _compute_slice_frame(analysis: pd.DataFrame) -> pd.DataFrame:
    score_edges = np.quantile(
        analysis["reference_pred"].astype("float64").to_numpy(),
        q=np.linspace(0.0, 1.0, 11),
    )
    score_edges[0] = 0.0
    score_edges[-1] = 1.0
    confidence_edges = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    disagreement_edges = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 1.0]

    analysis = analysis.copy()
    analysis["v3_score_decile"] = _bucket_from_edges(analysis["reference_pred"], score_edges, prefix="score")
    analysis["v3_confidence_bucket"] = _bucket_from_edges(
        (analysis["reference_pred"].astype("float64") - 0.5).abs(),
        confidence_edges,
        prefix="confidence",
    )
    analysis["v3_disagreement_bucket"] = _bucket_from_edges(
        (analysis["candidate_pred"].astype("float64") - analysis["reference_pred"].astype("float64")).abs(),
        disagreement_edges,
        prefix="disagreement",
    )
    return analysis


def _aggregate_family_delta(
    analysis: pd.DataFrame,
    *,
    family_column: str,
    challenger_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_rows = float(len(analysis))
    for family_value, part in analysis.groupby(family_column, dropna=False):
        y_true = part[OOF_TARGET_COLUMN].astype(int)
        reference_auc = _safe_binary_auc(y_true, part["reference_pred"])
        candidate_auc = _safe_binary_auc(y_true, part["candidate_pred"])
        reference_logloss = float(part["reference_loss"].mean())
        candidate_logloss = float(part["candidate_loss"].mean())
        rows.append(
            {
                "challenger": challenger_name,
                "family_level": family_column,
                "family_value": str(family_value),
                "rows": int(len(part)),
                "reference_auc": reference_auc,
                "candidate_auc": candidate_auc,
                "delta_auc": float(candidate_auc - reference_auc),
                "reference_logloss": reference_logloss,
                "candidate_logloss": candidate_logloss,
                "delta_logloss": float(candidate_logloss - reference_logloss),
                "reference_logloss_contribution": float(reference_logloss * len(part) / total_rows),
                "candidate_logloss_contribution": float(candidate_logloss * len(part) / total_rows),
                "delta_logloss_contribution": float((candidate_logloss - reference_logloss) * len(part) / total_rows),
                "v3_wins_auc": bool(reference_auc >= candidate_auc),
                "v3_wins_logloss": bool(reference_logloss <= candidate_logloss),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_slice_delta(
    analysis: pd.DataFrame,
    *,
    slice_column: str,
    challenger_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slice_value, part in analysis.groupby(slice_column, dropna=False):
        y_true = part[OOF_TARGET_COLUMN].astype(int)
        reference_auc = _safe_binary_auc(y_true, part["reference_pred"])
        candidate_auc = _safe_binary_auc(y_true, part["candidate_pred"])
        rows.append(
            {
                "challenger": challenger_name,
                "slice_type": slice_column,
                "slice_value": str(slice_value),
                "rows": int(len(part)),
                "reference_auc": reference_auc,
                "candidate_auc": candidate_auc,
                "delta_auc": float(candidate_auc - reference_auc) if np.isfinite(reference_auc) and np.isfinite(candidate_auc) else float("nan"),
                "reference_logloss": float(part["reference_loss"].mean()),
                "candidate_logloss": float(part["candidate_loss"].mean()),
                "delta_logloss": float(part["candidate_loss"].mean() - part["reference_loss"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_one_challenger_summary(
    *,
    analysis: pd.DataFrame,
    challenger: ChallengerSpec,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_true = analysis[OOF_TARGET_COLUMN].astype(int)
    reference_auc = float(binary_auc(y_true, analysis["reference_pred"]))
    candidate_auc = float(binary_auc(y_true, analysis["candidate_pred"]))
    reference_logloss = float(analysis["reference_loss"].mean())
    candidate_logloss = float(analysis["candidate_loss"].mean())

    segment3_delta = _aggregate_family_delta(analysis, family_column="segment3", challenger_name=challenger.name)
    segment5_delta = _aggregate_family_delta(analysis, family_column="segment5", challenger_name=challenger.name)
    slices = pd.concat(
        [
            _aggregate_slice_delta(analysis, slice_column="v3_score_decile", challenger_name=challenger.name),
            _aggregate_slice_delta(analysis, slice_column="v3_confidence_bucket", challenger_name=challenger.name),
            _aggregate_slice_delta(analysis, slice_column="v3_disagreement_bucket", challenger_name=challenger.name),
        ],
        axis=0,
        ignore_index=True,
    )

    dominant_segment3 = segment3_delta.loc[segment3_delta["family_value"].eq(DOMINANT_MACROFAMILY)]
    dominant_payload = dominant_segment3.iloc[0].to_dict() if not dominant_segment3.empty else None
    summary = {
        "challenger": challenger.name,
        "challenger_family": challenger.family,
        "global_rows": int(len(analysis)),
        "reference_auc": reference_auc,
        "candidate_auc": candidate_auc,
        "delta_auc": float(candidate_auc - reference_auc),
        "reference_logloss": reference_logloss,
        "candidate_logloss": candidate_logloss,
        "delta_logloss": float(candidate_logloss - reference_logloss),
        "pearson_corr_vs_v3": float(analysis["reference_pred"].corr(analysis["candidate_pred"])),
        "mean_abs_pred_diff_vs_v3": float((analysis["candidate_pred"] - analysis["reference_pred"]).abs().mean()),
        "dominant_segment3": dominant_payload,
        "segment3_v3_auc_win_rate": float(segment3_delta["v3_wins_auc"].mean()),
        "segment3_v3_logloss_win_rate": float(segment3_delta["v3_wins_logloss"].mean()),
        "segment5_v3_auc_win_rate": float(segment5_delta["v3_wins_auc"].mean()),
        "segment5_v3_logloss_win_rate": float(segment5_delta["v3_wins_logloss"].mean()),
        "top_segment3_damage_losses": segment3_delta.sort_values(
            by=["delta_logloss_contribution", "delta_logloss"],
            ascending=False,
        ).head(10).to_dict(orient="records"),
        "top_segment3_damage_gains": segment3_delta.sort_values(
            by=["delta_logloss_contribution", "delta_logloss"],
            ascending=True,
        ).head(10).to_dict(orient="records"),
        "top_segment5_damage_losses": segment5_delta.sort_values(
            by=["delta_logloss_contribution", "delta_logloss"],
            ascending=False,
        ).head(10).to_dict(orient="records"),
        "top_disagreement_losses": slices.loc[slices["slice_type"].eq("v3_disagreement_bucket")].sort_values(
            by=["delta_logloss", "delta_auc"],
            ascending=[False, True],
        ).head(10).to_dict(orient="records"),
    }
    return summary, pd.concat([segment3_delta, segment5_delta], axis=0, ignore_index=True), slices


def _derive_operational_recommendation(challenger_summaries: pd.DataFrame) -> str:
    if challenger_summaries.empty:
        return "No hay challengers para analizar."
    dominant_losses = challenger_summaries["dominant_delta_logloss"].astype("float64")
    global_losses = challenger_summaries["delta_logloss"].astype("float64")
    if bool((dominant_losses > 0.0).all() and (global_losses > 0.0).all()):
        return (
            "v3 gana de forma consistente por mejor ranking/logloss en la macrofamilia dominante y tambien fuera de ella; "
            "la siguiente hipotesis no debe ser otro challenger standalone, sino una explicacion/regularizacion de por que "
            "el blend residual actual conserva mejor el ranking global."
        )
    return (
        "La superioridad de v3 no es completamente uniforme; revisar los slices donde algun challenger gane logloss o AUC "
        "antes de abrir la siguiente hipotesis."
    )


def _derive_dominance_patterns(
    challenger_summary: pd.DataFrame,
    slice_delta: pd.DataFrame,
) -> dict[str, Any]:
    patterns: dict[str, Any] = {}
    if challenger_summary.empty:
        return patterns

    dominant_losses = challenger_summary["dominant_delta_logloss"].astype("float64")
    if bool((dominant_losses > 0.0).all()):
        patterns["dominant_family_consistent_v3_advantage"] = True

    confidence = slice_delta.loc[slice_delta["slice_type"].eq("v3_confidence_bucket")].copy()
    if not confidence.empty:
        grouped = (
            confidence.groupby("slice_value", dropna=False)
            .agg(
                mean_delta_logloss=("delta_logloss", "mean"),
                challenger_count=("challenger", "nunique"),
                support_rows=("rows", "sum"),
            )
            .reset_index()
            .sort_values(by=["mean_delta_logloss", "support_rows"], ascending=[False, False])
        )
        patterns["confidence_slice_pressure"] = grouped.head(5).to_dict(orient="records")

    disagreement = slice_delta.loc[slice_delta["slice_type"].eq("v3_disagreement_bucket")].copy()
    if not disagreement.empty:
        grouped = (
            disagreement.groupby("slice_value", dropna=False)
            .agg(
                mean_delta_logloss=("delta_logloss", "mean"),
                challenger_count=("challenger", "nunique"),
                support_rows=("rows", "sum"),
            )
            .reset_index()
            .sort_values(by=["mean_delta_logloss", "support_rows"], ascending=[False, False])
        )
        patterns["disagreement_slice_pressure"] = grouped.head(5).to_dict(orient="records")

    return patterns


def _derive_specific_recommendation(
    challenger_summary: pd.DataFrame,
    slice_delta: pd.DataFrame,
) -> str:
    if challenger_summary.empty:
        return "No hay suficiente informacion para proponer la siguiente hipotesis."

    dominant_consistent = bool((challenger_summary["dominant_delta_logloss"].astype("float64") > 0.0).mean() >= 0.8)
    confidence = slice_delta.loc[slice_delta["slice_type"].eq("v3_confidence_bucket")].copy()
    disagreement = slice_delta.loc[slice_delta["slice_type"].eq("v3_disagreement_bucket")].copy()

    low_confidence_worse = False
    if not confidence.empty:
        low_confidence = confidence.loc[confidence["slice_value"].isin(["confidence_00", "confidence_01", "confidence_02"])]
        low_confidence_worse = bool((low_confidence.groupby("challenger")["delta_logloss"].mean() > 0.0).all())

    high_disagreement_worse = False
    if not disagreement.empty:
        high_disagreement = disagreement.loc[disagreement["slice_value"].isin(["disagreement_03", "disagreement_04", "disagreement_05"])]
        high_disagreement_worse = bool((high_disagreement.groupby("challenger")["delta_logloss"].mean() > 0.0).all())

    if dominant_consistent and low_confidence_worse and high_disagreement_worse:
        return (
            "La siguiente hipotesis debe centrarse en la banda de baja confianza de v3 dentro de la macrofamilia dominante: "
            "no otro challenger global, sino una correccion que solo intente reordenar ejemplos de alta ambiguedad "
            "(`abs(v3-0.5) <= 0.20`) y alto desacuerdo contra v3 dentro de `Electronic check__Month-to-month__Fiber optic`. "
            "La excepcion parcial es el residual casi-vivo, que mejora algo de logloss pero no ranking; eso sugiere atacar la zona ambigua, "
            "no reemplazar a v3 globalmente."
        )

    return _derive_operational_recommendation(challenger_summary)


def run_v3_dominance_diagnostic(
    *,
    train_csv_path: str,
    reference_oof_spec: str,
    challengers: Sequence[ChallengerSpec],
    out_dir: str | Path,
    label: str = "v3_dominance",
) -> dict[str, Any]:
    """Compare v3 against a curated challenger set using only existing OOF artifacts."""
    out_root = Path(out_dir)
    train_df = load_csv(train_csv_path)
    family_frame = build_family_frame(train_df)
    reference_frame, reference_source = load_reference_prediction_frame(
        reference_oof_spec=reference_oof_spec,
        id_column=ID_COLUMN,
        target_column=OOF_TARGET_COLUMN,
    )
    reference_analysis = reference_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "reference_pred"]].merge(
        family_frame,
        how="inner",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if len(reference_analysis) != len(train_df):
        raise ValueError(
            f"Reference OOF does not align with train.csv row count: {len(reference_analysis)} vs {len(train_df)}"
        )

    all_family_rows: list[pd.DataFrame] = []
    all_slice_rows: list[pd.DataFrame] = []
    challenger_summary_rows: list[dict[str, Any]] = []

    for challenger in challengers:
        candidate_frame = _load_candidate_frame(challenger)
        merged = _validate_pair(reference_frame, candidate_frame, challenger_name=challenger.name)
        analysis = merged.merge(
            family_frame,
            how="inner",
            on=ID_COLUMN,
            validate="one_to_one",
        )
        analysis["reference_pred"] = _clip_probability(analysis["reference_pred"])
        analysis["candidate_pred"] = _clip_probability(analysis["candidate_pred"])
        analysis["reference_loss"] = _logloss_vector(analysis[OOF_TARGET_COLUMN], analysis["reference_pred"])
        analysis["candidate_loss"] = _logloss_vector(analysis[OOF_TARGET_COLUMN], analysis["candidate_pred"])
        analysis = _compute_slice_frame(analysis)

        summary, family_rows, slice_rows = _build_one_challenger_summary(
            analysis=analysis,
            challenger=challenger,
        )
        dominant_segment3 = summary.get("dominant_segment3") or {}
        challenger_summary_rows.append(
            {
                "challenger": challenger.name,
                "challenger_family": challenger.family,
                "delta_auc": float(summary["delta_auc"]),
                "delta_logloss": float(summary["delta_logloss"]),
                "pearson_corr_vs_v3": float(summary["pearson_corr_vs_v3"]),
                "mean_abs_pred_diff_vs_v3": float(summary["mean_abs_pred_diff_vs_v3"]),
                "dominant_delta_auc": float(dominant_segment3.get("delta_auc", np.nan)),
                "dominant_delta_logloss": float(dominant_segment3.get("delta_logloss", np.nan)),
                "segment3_v3_auc_win_rate": float(summary["segment3_v3_auc_win_rate"]),
                "segment3_v3_logloss_win_rate": float(summary["segment3_v3_logloss_win_rate"]),
                "segment5_v3_auc_win_rate": float(summary["segment5_v3_auc_win_rate"]),
                "segment5_v3_logloss_win_rate": float(summary["segment5_v3_logloss_win_rate"]),
            }
        )
        all_family_rows.append(family_rows)
        all_slice_rows.append(slice_rows)

    challenger_summary = pd.DataFrame(challenger_summary_rows).sort_values(
        by=["delta_auc", "delta_logloss"],
        ascending=[False, True],
    ).reset_index(drop=True)
    family_delta = pd.concat(all_family_rows, axis=0, ignore_index=True) if all_family_rows else pd.DataFrame()
    slice_delta = pd.concat(all_slice_rows, axis=0, ignore_index=True) if all_slice_rows else pd.DataFrame()

    summary_json = {
        "generated_at_utc": utc_now_iso(),
        "train_csv_path": str(train_csv_path),
        "reference_oof_spec": str(reference_oof_spec),
        "reference_source": reference_source,
        "reference_file": describe_file(Path(reference_oof_spec.split("#", 1)[0])),
        "challengers": [vars(item) for item in challengers],
        "dominant_family_value": DOMINANT_MACROFAMILY,
        "challenger_summary": challenger_summary.to_dict(orient="records"),
        "dominance_patterns": _derive_dominance_patterns(challenger_summary, slice_delta),
        "operational_recommendation": _derive_specific_recommendation(challenger_summary, slice_delta),
    }

    summary_path = ensure_parent_dir(out_root / f"{label}_summary.json")
    challenger_csv_path = ensure_parent_dir(out_root / f"{label}_challengers.csv")
    family_csv_path = ensure_parent_dir(out_root / f"{label}_families.csv")
    slice_csv_path = ensure_parent_dir(out_root / f"{label}_slices.csv")

    challenger_summary.to_csv(challenger_csv_path, index=False)
    family_delta.to_csv(family_csv_path, index=False)
    slice_delta.to_csv(slice_csv_path, index=False)
    write_json(summary_path, summary_json)

    return {
        "summary_path": str(summary_path),
        "challenger_csv_path": str(challenger_csv_path),
        "family_csv_path": str(family_csv_path),
        "slice_csv_path": str(slice_csv_path),
        "summary": summary_json,
    }
