#!/usr/bin/env python3
"""Generate a submission by chaining trained residual rerankers over a base submission."""

from __future__ import annotations

import argparse
import json
import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.diagnostics import parse_feature_blocks_arg
from churn_baseline.specialist import make_residual_reranker_chain_submission


def _parse_step(raw: str) -> dict[str, object]:
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) != 4:
        raise ValueError(
            "Each --step must use format "
            "'<preset>|<model_path>|<alpha>|<feature_blocks>'"
        )
    preset, model_path, alpha_raw, feature_blocks_raw = parts
    return {
        "preset": preset,
        "model_path": model_path,
        "alpha": float(alpha_raw),
        "feature_blocks": parse_feature_blocks_arg(feature_blocks_raw),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a submission by chaining residual rerankers over a reference submission"
    )
    parser.add_argument(
        "--train-csv",
        default="data/raw/train.csv",
        help="Path to train.csv for fit-aware feature blocks such as G.",
    )
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument(
        "--reference-submission",
        default="artifacts/submissions/playground-series-s6e3-rvblend.csv",
        help="Base submission CSV used as teacher/reference score.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/submissions/playground-series-s6e3-residual-hier.csv",
        help="Final submission CSV path.",
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/reports/submission_candidate_residual_hierarchical.json",
        help="Report JSON path.",
    )
    parser.add_argument(
        "--reference-component-csv",
        default="",
        help="Optional CSV with id + pred_* columns for teacher-aware rerankers.",
    )
    parser.add_argument(
        "--reference-mode",
        default="previous",
        choices=["previous", "base"],
        help="Use the previous step output or the base reference submission as the step teacher.",
    )
    parser.add_argument(
        "--step",
        action="append",
        required=True,
        help="Residual step spec: <preset>|<model_path>|<alpha>|<feature_blocks>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = [_parse_step(raw) for raw in args.step]
    reference_component_frame = pd.read_csv(args.reference_component_csv) if args.reference_component_csv.strip() else None
    if reference_component_frame is not None and args.reference_mode != "base":
        raise ValueError("--reference-component-csv requires --reference-mode base for teacher-aware parity")
    report = make_residual_reranker_chain_submission(
        test_csv_path=args.test_csv,
        train_csv_path=args.train_csv,
        reference_submission_path=args.reference_submission,
        reference_component_frame=reference_component_frame,
        reference_mode=args.reference_mode,
        steps=steps,
        output_csv_path=args.output_csv,
        report_path=args.report_json,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
