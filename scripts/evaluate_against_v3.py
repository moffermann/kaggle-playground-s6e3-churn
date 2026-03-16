#!/usr/bin/env python3
"""Evaluate a residual-chain challenger directly against a tracked incumbent."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.incumbent_v3 import evaluate_candidate_chain_against_incumbent
from churn_baseline.validation_protocol import DOMINANT_MACROFAMILY, SUPPORTED_STAGES


def _parse_candidate_step(raw: str) -> tuple[str, str]:
    parts = [part.strip() for part in raw.split("|", maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Each --candidate-step must use '<preset>|<oof_path>'")
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a residual-chain challenger directly against a tracked incumbent"
    )
    parser.add_argument("--incumbent", choices=["v3", "v6"], default="v3")
    parser.add_argument("--stage", choices=SUPPORTED_STAGES, default="smoke")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument(
        "--candidate-order",
        required=True,
        help="Comma-separated step order for the challenger chain.",
    )
    parser.add_argument(
        "--candidate-step",
        action="append",
        default=[],
        help="Optional challenger step OOF override: <preset>|<oof_path>.",
    )
    parser.add_argument(
        "--target-family-level",
        choices=["segment3", "segment5"],
        default="segment3",
    )
    parser.add_argument(
        "--target-family-value",
        default=DOMINANT_MACROFAMILY,
        help="Family key used as the target-family guardrail.",
    )
    parser.add_argument(
        "--dominant-family-value",
        default=DOMINANT_MACROFAMILY,
        help="Dominant macrofamily key for Split B.",
    )
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--out-dir", default="artifacts/reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_order = [token.strip() for token in args.candidate_order.split(",") if token.strip()]
    candidate_step_oof_paths = dict(_parse_candidate_step(raw) for raw in args.candidate_step)
    summary = evaluate_candidate_chain_against_incumbent(
        incumbent=args.incumbent,
        candidate_order=candidate_order,
        candidate_step_oof_paths=candidate_step_oof_paths,
        stage=args.stage,
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        target_family_level=args.target_family_level,
        target_family_value=args.target_family_value,
        dominant_family_value=args.dominant_family_value,
        label=args.label,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
