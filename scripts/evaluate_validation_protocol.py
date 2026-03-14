#!/usr/bin/env python3
"""Evaluate a candidate OOF artifact under the validation reset protocol."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import ID_COLUMN
from churn_baseline.diagnostics import DEFAULT_REFERENCE_OOF_SPECS
from churn_baseline.diagnostics import OOF_TARGET_COLUMN
from churn_baseline.validation_protocol import (
    DEFAULT_SUBMISSION_FORENSICS_SUMMARY_JSON,
    DOMINANT_MACROFAMILY,
    SUPPORTED_STAGES,
    SUPPORTED_TARGET_LEVELS,
    evaluate_validation_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a candidate under the validation reset protocol")
    parser.add_argument("--stage", choices=SUPPORTED_STAGES, default="smoke", help="Protocol stage to evaluate.")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--analysis-oof",
        default="",
        help=(
            "Optional prebuilt analysis OOF CSV containing "
            f"{ID_COLUMN},{OOF_TARGET_COLUMN},reference_pred,candidate_pred."
        ),
    )
    parser.add_argument(
        "--candidate-oof",
        default="",
        help="Optional candidate OOF spec: <path>[#<prediction_column>]. Used when --analysis-oof is not provided.",
    )
    parser.add_argument(
        "--reference-oof",
        default="",
        help="Optional direct reference OOF spec: <path>[#<prediction_column>]. Overrides --oof + --reference-weights-json.",
    )
    parser.add_argument(
        "--oof",
        action="append",
        default=[],
        help="OOF input spec for weighted reference: <name>=<path>[#<prediction_column>]. Defaults to cb/xgb/lgb/r/rv.",
    )
    parser.add_argument(
        "--reference-weights-json",
        default="artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json",
        help="Weights JSON for the default weighted reference.",
    )
    parser.add_argument(
        "--target-family-level",
        choices=SUPPORTED_TARGET_LEVELS,
        default="segment5",
        help="Family granularity for the declared target family.",
    )
    parser.add_argument("--target-family-value", default="", help="Target family key under the chosen family level.")
    parser.add_argument("--sister-family-value", default="", help="Optional explicit sister family override.")
    parser.add_argument(
        "--dominant-family-value",
        default=DOMINANT_MACROFAMILY,
        help="Dominant macrofamily key (segment3) used for the guardrail check.",
    )
    parser.add_argument("--candidate-metrics-json", default="", help="Optional candidate metrics JSON, used for midcap/submission cv_std checks.")
    parser.add_argument("--reference-metrics-json", default="", help="Optional reference metrics JSON, used for midcap/submission cv_std checks.")
    parser.add_argument("--submission-csv", default="", help="Optional submission CSV path, required for submission-stage artifact trace checks.")
    parser.add_argument(
        "--submission-forensics-summary",
        default=DEFAULT_SUBMISSION_FORENSICS_SUMMARY_JSON,
        help="Submission forensics summary JSON used by the submission-stage public-survival prior.",
    )
    parser.add_argument(
        "--submission-family",
        default="",
        help="Optional submission_family override for the submission-stage public-survival prior.",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/validation_protocol_verdict.json",
        help="Output JSON verdict path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    oof_specs = args.oof if args.oof else list(DEFAULT_REFERENCE_OOF_SPECS)
    result = evaluate_validation_protocol(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        stage=args.stage,
        analysis_oof_path=args.analysis_oof.strip() or None,
        candidate_oof_spec=args.candidate_oof.strip() or None,
        reference_oof_spec=args.reference_oof.strip() or None,
        oof_specs=oof_specs,
        reference_weights_json=args.reference_weights_json,
        target_family_level=args.target_family_level,
        target_family_value=args.target_family_value.strip() or None,
        sister_family_value=args.sister_family_value.strip() or None,
        dominant_family_value=args.dominant_family_value.strip() or DOMINANT_MACROFAMILY,
        candidate_metrics_json=args.candidate_metrics_json.strip() or None,
        reference_metrics_json=args.reference_metrics_json.strip() or None,
        submission_csv_path=args.submission_csv.strip() or None,
        submission_forensics_summary_json=args.submission_forensics_summary.strip() or None,
        submission_family_override=args.submission_family.strip() or None,
        out_json_path=args.out_json,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
