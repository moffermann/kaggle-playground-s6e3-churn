"""Ablation and compression analysis for the incumbent residual hierarchy v3."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .artifacts import ensure_parent_dir, write_json
from .diagnostics import utc_now_iso
from .incumbent_v3 import V3_ORDER, evaluate_candidate_chain_against_v3
from .validation_protocol import DOMINANT_MACROFAMILY, MIDCAP_STAGE, SUPPORTED_STAGES


DEFAULT_ABLATION_OUT_DIR = "artifacts/reports/residual_ablation_v1"
COMPRESSION_TOLERANCES: tuple[float, ...] = (1e-05, 5e-05, 1e-04, 5e-04, 1e-03)


def _iter_ordered_subsequences(
    order: Sequence[str],
    *,
    include_full: bool = False,
) -> list[tuple[str, ...]]:
    items = tuple(str(step) for step in order)
    min_size = 1
    max_size = len(items) if include_full else len(items) - 1
    out: list[tuple[str, ...]] = []
    for size in range(min_size, max_size + 1):
        out.extend(tuple(items[idx] for idx in combo) for combo in combinations(range(len(items)), size))
    return out


def _make_label(order: Sequence[str]) -> str:
    return "residual_ablation_" + "__".join(str(step) for step in order)


def _dropped_steps(reference_order: Sequence[str], candidate_order: Sequence[str]) -> list[str]:
    candidate_set = set(candidate_order)
    return [str(step) for step in reference_order if str(step) not in candidate_set]


def _candidate_kind(reference_order: Sequence[str], candidate_order: Sequence[str]) -> str:
    dropped = len(reference_order) - len(candidate_order)
    if dropped == 1:
        return "single_drop"
    if dropped == 2:
        return "pair_drop"
    if dropped >= 3:
        return "compressed"
    return "full"


def _read_protocol_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _best_row_payload(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    row = frame.sort_values(["delta_vs_v3_oof_auc", "chain_size"], ascending=[False, True]).iloc[0]
    return {
        "label": str(row["label"]),
        "candidate_order": list(row["candidate_order"]),
        "dropped_steps": list(row["dropped_steps"]),
        "chain_size": int(row["chain_size"]),
        "delta_vs_v3_oof_auc": float(row["delta_vs_v3_oof_auc"]),
        "verdict": str(row["verdict"]),
        "dominant_family_delta_auc": None if pd.isna(row["dominant_family_delta_auc"]) else float(row["dominant_family_delta_auc"]),
        "target_family_delta_auc": None if pd.isna(row["target_family_delta_auc"]) else float(row["target_family_delta_auc"]),
    }


def _build_recommendation(*, best_compressed_delta: float, gate_delta: float) -> str:
    if best_compressed_delta >= gate_delta:
        return (
            "There is a promotable compressed residual chain. Review the best compressed candidate and verify whether "
            "the public family logic still holds before replacing v3."
        )
    if best_compressed_delta >= 0.0:
        return (
            "The residual hierarchy is partially compressible without losing global AUC, but not enough to clear the "
            "current promotion gate. Use the best compressed chain for reasoning, not for replacement."
        )
    return (
        "The residual hierarchy is not compressible under the current gate. v3 retains edge across its full chain, so "
        "future work should treat the residual family as the incumbent core rather than as accidental complexity."
    )


def run_residual_hierarchy_ablation(
    *,
    train_csv_path: str | Path = "data/raw/train.csv",
    test_csv_path: str | Path = "data/raw/test.csv",
    stage: str = MIDCAP_STAGE,
    target_family_level: str = "segment3",
    target_family_value: str = DOMINANT_MACROFAMILY,
    dominant_family_value: str = DOMINANT_MACROFAMILY,
    out_dir: str | Path = DEFAULT_ABLATION_OUT_DIR,
) -> dict[str, Any]:
    stage_normalized = str(stage).strip().lower()
    if stage_normalized not in SUPPORTED_STAGES:
        raise ValueError(f"Unsupported stage '{stage}'. Supported: {SUPPORTED_STAGES}")

    out_root = Path(out_dir)
    candidates = _iter_ordered_subsequences(V3_ORDER, include_full=False)
    rows: list[dict[str, Any]] = []

    for candidate_order in candidates:
        label = _make_label(candidate_order)
        summary = evaluate_candidate_chain_against_v3(
            candidate_order=candidate_order,
            stage=stage_normalized,
            train_csv_path=train_csv_path,
            test_csv_path=test_csv_path,
            target_family_level=target_family_level,
            target_family_value=target_family_value,
            dominant_family_value=dominant_family_value,
            label=label,
            out_dir=out_root,
        )
        protocol_payload = _read_protocol_payload(summary["verdict_path"])
        dropped_steps = _dropped_steps(V3_ORDER, candidate_order)
        rows.append(
            {
                "label": str(label),
                "candidate_order": list(candidate_order),
                "chain_size": int(len(candidate_order)),
                "dropped_steps": dropped_steps,
                "dropped_count": int(len(dropped_steps)),
                "candidate_kind": _candidate_kind(V3_ORDER, candidate_order),
                "delta_vs_v3_oof_auc": float(summary["delta_vs_v3_oof_auc"]),
                "candidate_oof_auc": float(summary["overall_metrics"]["candidate_oof_auc"]),
                "reference_oof_auc": float(summary["overall_metrics"]["reference_oof_auc"]),
                "candidate_cv_std": float(summary["overall_metrics"]["candidate_cv_std"]),
                "reference_cv_std": float(summary["overall_metrics"]["reference_cv_std"]),
                "verdict": str(summary["verdict"]["overall_status"]),
                "dominant_family_delta_auc": None
                if protocol_payload.get("dominant_family") is None
                else protocol_payload["dominant_family"].get("delta_auc"),
                "target_family_delta_auc": None
                if protocol_payload.get("target_family") is None
                else protocol_payload["target_family"].get("delta_auc"),
                "summary_path": str(summary["summary_path"]),
                "verdict_path": str(summary["verdict_path"]),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["delta_vs_v3_oof_auc", "chain_size", "label"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    single_drop = results.loc[results["dropped_count"].eq(1)].copy()
    if not single_drop.empty:
        single_drop["removed_step"] = single_drop["dropped_steps"].map(lambda items: items[0])
        single_drop["marginal_edge_lost"] = -single_drop["delta_vs_v3_oof_auc"].astype(float)
        single_drop = single_drop.sort_values("marginal_edge_lost", ascending=False).reset_index(drop=True)

    pair_drop = results.loc[results["dropped_count"].eq(2)].copy()
    if not pair_drop.empty:
        pair_drop["removed_pair"] = pair_drop["dropped_steps"].map(lambda items: " + ".join(items))
        pair_drop["marginal_edge_lost"] = -pair_drop["delta_vs_v3_oof_auc"].astype(float)
        pair_drop = pair_drop.sort_values("marginal_edge_lost", ascending=False).reset_index(drop=True)

    best_by_size = []
    for chain_size, part in results.groupby("chain_size", dropna=False):
        payload = _best_row_payload(part)
        if payload is not None:
            best_by_size.append(payload)

    compressed = results.loc[results["chain_size"].lt(len(V3_ORDER))].copy()
    compression_thresholds = []
    for tolerance in COMPRESSION_TOLERANCES:
        eligible = compressed.loc[compressed["delta_vs_v3_oof_auc"].ge(-float(tolerance))]
        best_payload = _best_row_payload(eligible)
        compression_thresholds.append(
            {
                "tolerance_abs_delta_auc": float(tolerance),
                "has_eligible_chain": bool(best_payload is not None),
                "best_chain": best_payload,
            }
        )

    best_compressed = _best_row_payload(compressed)
    best_compressed_delta = float(best_compressed["delta_vs_v3_oof_auc"]) if best_compressed else float("-inf")
    summary = {
        "generated_at_utc": utc_now_iso(),
        "stage": stage_normalized,
        "reference_order": list(V3_ORDER),
        "evaluated_candidate_count": int(len(results)),
        "best_compressed_chain": best_compressed,
        "best_chain_by_size": best_by_size,
        "single_step_ablation": []
        if single_drop.empty
        else single_drop[
            [
                "removed_step",
                "delta_vs_v3_oof_auc",
                "marginal_edge_lost",
                "verdict",
                "dominant_family_delta_auc",
                "target_family_delta_auc",
            ]
        ].to_dict(orient="records"),
        "pair_step_ablation": []
        if pair_drop.empty
        else pair_drop[
            [
                "removed_pair",
                "delta_vs_v3_oof_auc",
                "marginal_edge_lost",
                "verdict",
                "dominant_family_delta_auc",
                "target_family_delta_auc",
            ]
        ].to_dict(orient="records"),
        "compression_thresholds": compression_thresholds,
        "recommendation": _build_recommendation(best_compressed_delta=best_compressed_delta, gate_delta=1e-05),
    }

    summary_path = ensure_parent_dir(out_root / "residual_ablation_summary.json")
    csv_path = ensure_parent_dir(out_root / "residual_ablation_candidates.csv")
    write_json(summary_path, summary)
    results.to_csv(csv_path, index=False)
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "candidates_csv_path": str(csv_path),
    }
