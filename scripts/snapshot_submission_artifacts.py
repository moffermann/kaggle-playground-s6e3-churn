#!/usr/bin/env python3
"""Capture reproducibility snapshot for a submission candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.diagnostics import (
    collect_git_context,
    describe_file,
    load_json_if_exists,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot submission artifacts with hashes")
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/train_cv_multiseed_metrics.json",
        help="Training metrics JSON used by submission.",
    )
    parser.add_argument(
        "--submission-csv",
        default="",
        help="Submission CSV to snapshot.",
    )
    parser.add_argument(
        "--weights-json-path",
        default="",
        help="Optional weights/config JSON used for blend/stack submissions.",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Additional artifact path to include (repeatable).",
    )
    parser.add_argument("--label", default="", help="Optional human-readable snapshot label.")
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/submission_artifact_snapshot.json",
        help="Output snapshot JSON path.",
    )
    return parser.parse_args()


def _extract_model_paths(metrics: dict[str, object]) -> list[str]:
    paths: list[str] = []
    maybe_single = metrics.get("model_path")
    if isinstance(maybe_single, str):
        paths.append(maybe_single)
    maybe_multi = metrics.get("model_paths")
    if isinstance(maybe_multi, list):
        for item in maybe_multi:
            if isinstance(item, str):
                paths.append(item)
    deduped: list[str] = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def main() -> int:
    args = parse_args()

    metrics_payload = load_json_if_exists(args.metrics_path)
    model_paths = _extract_model_paths(metrics_payload or {})

    artifacts: dict[str, dict[str, object]] = {}
    artifacts["metrics"] = describe_file(args.metrics_path)
    if args.submission_csv.strip():
        artifacts["submission"] = describe_file(args.submission_csv)
    if args.weights_json_path.strip():
        artifacts["weights"] = describe_file(args.weights_json_path)

    for idx, model_path in enumerate(model_paths, start=1):
        artifacts[f"model_{idx}"] = describe_file(model_path)

    for idx, path in enumerate(args.extra_path, start=1):
        artifacts[f"extra_{idx}"] = describe_file(path)

    summary = {
        "generated_at_utc": utc_now_iso(),
        "label": args.label or None,
        "git": collect_git_context(),
        "inputs": {
            "metrics_path": args.metrics_path,
            "submission_csv": args.submission_csv or None,
            "weights_json_path": args.weights_json_path or None,
            "extra_paths": args.extra_path,
        },
        "metrics_summary": None,
        "artifacts": artifacts,
    }
    if metrics_payload is not None:
        summary["metrics_summary"] = {
            "feature_blocks": metrics_payload.get("feature_blocks"),
            "feature_count": metrics_payload.get("feature_count"),
            "oof_auc": metrics_payload.get("oof_auc"),
            "ensemble_oof_auc": metrics_payload.get("ensemble_oof_auc"),
            "cv_mean_auc": metrics_payload.get("cv_mean_auc"),
            "cv_std_auc": metrics_payload.get("cv_std_auc"),
            "model_paths_count": len(model_paths),
        }

    write_json(args.out_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
