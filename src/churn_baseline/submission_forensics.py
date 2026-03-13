"""Submission forensics: join local artifacts with Kaggle submission history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ensure_parent_dir, write_json
from .diagnostics import utc_now_iso
from .kaggle_api import build_authenticated_api


LOCAL_SUBMISSION_PREFIX = "artifacts/submissions/"
PRIMARY_METRIC_PRIORITY: tuple[str, ...] = (
    "delta_vs_v3_oof_auc",
    "delta_oof_auc",
    "delta_vs_reference_oof_auc",
    "candidate_oof_auc",
    "ensemble_oof_auc",
    "full_auc",
    "oof_auc",
    "cv_mean_auc",
    "holdout_auc",
)


def _normalize_submission_path(raw: str) -> str:
    normalized = str(raw).replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _iter_submission_strings(payload: Any) -> list[str]:
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
            return
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if not isinstance(value, str):
            return
        normalized = _normalize_submission_path(value)
        if normalized.lower().endswith(".csv") and LOCAL_SUBMISSION_PREFIX in normalized:
            found.append(normalized)

    walk(payload)
    return sorted(set(found))


def _flatten_numeric_leaves(payload: Any) -> dict[str, float]:
    flat: dict[str, float] = {}

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                walk(child, next_prefix)
            return
        if isinstance(value, list):
            for idx, child in enumerate(value):
                next_prefix = f"{prefix}[{idx}]"
                walk(child, next_prefix)
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            flat[prefix] = float(value)

    walk(payload, "")
    return flat


def _select_primary_metric(flat_metrics: dict[str, float]) -> tuple[str | None, float | None]:
    if not flat_metrics:
        return None, None
    for metric_name in PRIMARY_METRIC_PRIORITY:
        for key, value in flat_metrics.items():
            if key == metric_name or key.endswith(f".{metric_name}"):
                return metric_name, float(value)
    return None, None


def _report_priority(report_path: str, *, has_metric: bool) -> tuple[int, str]:
    name = Path(report_path).name.lower()
    if name.startswith("submission_candidate_"):
        bucket = 0
    elif name.startswith("candidate_submission_"):
        bucket = 1
    elif name.startswith("validation_protocol_"):
        bucket = 2
    elif "submission" in name:
        bucket = 3
    else:
        bucket = 4
    metric_penalty = 0 if has_metric else 1
    return (bucket, metric_penalty, name)


def _infer_submission_family(file_name: str) -> str:
    name = file_name.lower()
    if "residual-hier" in name:
        return "residual_hierarchy"
    if "rvblend" in name:
        return "teacher_blend_rv"
    if "rblend" in name:
        return "teacher_blend_r"
    if "blend-grid" in name:
        return "blend_scan"
    if "multiseed" in name:
        return "catboost_multiseed"
    if "pseudo" in name:
        return "pseudo_labeling"
    return "generic"


def _submission_to_dict(submission: Any) -> dict[str, Any]:
    if hasattr(submission, "to_dict"):
        payload = submission.to_dict()
        if isinstance(payload, dict):
            return payload
    return {
        "ref": getattr(submission, "ref", None),
        "totalBytes": getattr(submission, "total_bytes", None),
        "date": getattr(submission, "date", None),
        "description": getattr(submission, "description", None),
        "errorDescription": getattr(submission, "error_description", None),
        "fileName": getattr(submission, "file_name", None),
        "publicScore": getattr(submission, "public_score", None),
        "privateScore": getattr(submission, "private_score", None),
        "status": getattr(submission, "status", None),
        "submittedBy": getattr(submission, "submitted_by", None),
        "submittedByRef": getattr(submission, "submitted_by_ref", None),
        "teamName": getattr(submission, "team_name", None),
        "url": getattr(submission, "url", None),
    }


def fetch_kaggle_submission_history(competition: str) -> pd.DataFrame:
    """Return Kaggle submission history for the current account."""
    api = build_authenticated_api()
    raw_rows: list[dict[str, Any]] = []
    for submission in api.competition_submissions(competition):
        payload = _submission_to_dict(submission)
        raw_rows.append(
            {
                "ref": int(payload["ref"]),
                "file_name": str(payload.get("fileName") or ""),
                "date": str(payload.get("date") or ""),
                "description": str(payload.get("description") or ""),
                "status": str(payload.get("status") or ""),
                "public_score": None if payload.get("publicScore") in (None, "") else float(payload["publicScore"]),
                "private_score": None if payload.get("privateScore") in (None, "") else float(payload["privateScore"]),
            }
        )
    history = pd.DataFrame(raw_rows)
    if history.empty:
        return history
    history["date"] = pd.to_datetime(history["date"], utc=True, errors="coerce")
    history = history.sort_values("date", ascending=False).reset_index(drop=True)
    history["submission_family"] = history["file_name"].map(_infer_submission_family)
    history["local_submission_path"] = history["file_name"].map(lambda value: f"{LOCAL_SUBMISSION_PREFIX}{value}")
    return history


def scan_local_submission_reports(reports_dir: str | Path) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Scan report JSONs and map them to submission CSV references."""
    report_rows: list[dict[str, Any]] = []
    per_submission_name: dict[str, list[dict[str, Any]]] = {}

    for report_path in sorted(Path(reports_dir).glob("*.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        submission_paths = _iter_submission_strings(payload)
        if not submission_paths:
            continue
        flat_metrics = _flatten_numeric_leaves(payload)
        metric_name, metric_value = _select_primary_metric(flat_metrics)
        report_record = {
            "report_path": str(report_path).replace("\\", "/"),
            "report_name": report_path.name,
            "submission_paths": submission_paths,
            "primary_metric_name": metric_name,
            "primary_metric_value": metric_value,
            "all_metric_names": sorted(flat_metrics),
        }
        report_rows.append(
            {
                "report_path": report_record["report_path"],
                "report_name": report_record["report_name"],
                "linked_submission_count": int(len(submission_paths)),
                "primary_metric_name": metric_name,
                "primary_metric_value": metric_value,
            }
        )
        for submission_path in submission_paths:
            submission_name = Path(submission_path).name
            per_submission_name.setdefault(submission_name, []).append(report_record)

    return pd.DataFrame(report_rows), per_submission_name


def build_submission_forensics(
    *,
    competition: str,
    reports_dir: str | Path,
    submissions_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build a joined submission ledger and summary."""
    kaggle_history = fetch_kaggle_submission_history(competition)
    report_frame, report_map = scan_local_submission_reports(reports_dir)

    local_submission_files = sorted(str(path).replace("\\", "/") for path in Path(submissions_dir).glob("*.csv"))
    local_submission_names = {Path(path).name for path in local_submission_files}

    ledger_rows: list[dict[str, Any]] = []
    for row in kaggle_history.to_dict(orient="records"):
        linked_reports = report_map.get(str(row["file_name"]), [])
        primary_report = None
        if linked_reports:
            primary_report = sorted(
                linked_reports,
                key=lambda item: _report_priority(item["report_path"], has_metric=item["primary_metric_name"] is not None),
            )[0]
        ledger_rows.append(
            {
                **row,
                "has_local_submission_file": row["file_name"] in local_submission_names,
                "linked_report_count": int(len(linked_reports)),
                "primary_report_path": None if primary_report is None else primary_report["report_path"],
                "primary_report_name": None if primary_report is None else primary_report["report_name"],
                "local_primary_metric_name": None if primary_report is None else primary_report["primary_metric_name"],
                "local_primary_metric_value": None if primary_report is None else primary_report["primary_metric_value"],
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    if ledger.empty:
        summary = {
            "generated_at_utc": utc_now_iso(),
            "competition": competition,
            "history_rows": 0,
            "local_submission_files": int(len(local_submission_files)),
            "report_rows": int(len(report_frame)),
            "error": "No Kaggle submissions returned.",
        }
        return ledger, report_frame, summary

    complete_mask = ledger["status"].astype(str).str.contains("COMPLETE", case=False, regex=False)
    complete_scores = ledger.loc[complete_mask & ledger["public_score"].notna()].copy()
    best_row = None if complete_scores.empty else complete_scores.sort_values("public_score", ascending=False).iloc[0]

    correlation_rows = []
    for metric_name, part in ledger.groupby("local_primary_metric_name", dropna=True):
        subset = part.loc[part["public_score"].notna() & part["local_primary_metric_value"].notna()].copy()
        if len(subset) < 3:
            continue
        subset = subset.sort_values("date")
        public_score = subset["public_score"].astype(float)
        local_metric = subset["local_primary_metric_value"].astype(float)
        correlation_rows.append(
            {
                "metric_name": str(metric_name),
                "rows": int(len(subset)),
                "pearson": float(public_score.corr(local_metric, method="pearson")),
                "spearman": float(public_score.corr(local_metric, method="spearman")),
            }
        )
    if correlation_rows:
        correlation_frame = pd.DataFrame(correlation_rows).sort_values(["rows", "spearman"], ascending=[False, False])
    else:
        correlation_frame = pd.DataFrame(columns=["metric_name", "rows", "pearson", "spearman"])

    family_frame = (
        ledger.loc[complete_mask & ledger["public_score"].notna()]
        .groupby("submission_family", dropna=False)
        .agg(
            submissions=("file_name", "count"),
            best_public_score=("public_score", "max"),
            mean_public_score=("public_score", "mean"),
        )
        .reset_index()
        .sort_values(["best_public_score", "mean_public_score"], ascending=False)
    )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "competition": competition,
        "history_rows": int(len(ledger)),
        "complete_rows": int(complete_mask.sum()),
        "error_rows": int(ledger["status"].astype(str).str.contains("ERROR", case=False, regex=False).sum()),
        "local_submission_files": int(len(local_submission_files)),
        "report_rows": int(len(report_frame)),
        "matched_history_rows": int((ledger["linked_report_count"] > 0).sum()),
        "matched_with_local_metric_rows": int(ledger["local_primary_metric_name"].notna().sum()),
        "best_public_submission": None
        if best_row is None
        else {
            "file_name": str(best_row["file_name"]),
            "ref": int(best_row["ref"]),
            "public_score": float(best_row["public_score"]),
            "submission_family": str(best_row["submission_family"]),
        },
        "family_summary": family_frame.to_dict(orient="records"),
        "local_public_correlation": correlation_frame.to_dict(orient="records"),
        "recent_complete_submissions": ledger.loc[complete_mask, ["file_name", "ref", "date", "public_score", "submission_family"]]
        .head(10)
        .assign(date=lambda df: df["date"].astype(str))
        .to_dict(orient="records"),
        "recommendation": (
            "Use v3 as incumbent and distrust tiny local deltas unless the same family has already shown "
            "public survival. Submission history is clustered by family, so forensics should gate any new line "
            "before CPU is spent on midcap."
        ),
    }
    return ledger, report_frame, summary


def write_submission_forensics_outputs(
    *,
    competition: str,
    reports_dir: str | Path,
    submissions_dir: str | Path,
    out_summary_json: str | Path,
    out_ledger_csv: str | Path,
    out_reports_csv: str | Path,
) -> dict[str, Any]:
    ledger, report_frame, summary = build_submission_forensics(
        competition=competition,
        reports_dir=reports_dir,
        submissions_dir=submissions_dir,
    )
    ensure_parent_dir(out_ledger_csv)
    ensure_parent_dir(out_reports_csv)
    ledger_to_write = ledger.copy()
    if "date" in ledger_to_write.columns:
        ledger_to_write["date"] = ledger_to_write["date"].astype(str)
    ledger_to_write.to_csv(out_ledger_csv, index=False)
    report_frame.to_csv(out_reports_csv, index=False)
    write_json(out_summary_json, summary)
    return summary
