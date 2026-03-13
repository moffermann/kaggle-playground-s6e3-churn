#!/usr/bin/env python3
"""Analyze why incumbent v3 dominates a curated challenger set."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.v3_dominance import (
    DEFAULT_CHALLENGERS,
    DEFAULT_V3_OOF_SPEC,
    ChallengerSpec,
    run_v3_dominance_diagnostic,
)


def _parse_challenger(raw: str) -> ChallengerSpec:
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) not in {2, 3, 4}:
        raise ValueError(
            "Each --challenger must use '<name>|<path>' or '<name>|<path>|<prediction_column>|<family>'"
        )
    name = parts[0]
    path = parts[1]
    prediction_column = parts[2] if len(parts) >= 3 and parts[2] else "oof_pred"
    family = parts[3] if len(parts) >= 4 and parts[3] else "custom"
    return ChallengerSpec(name=name, path=path, prediction_column=prediction_column, family=family)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze why incumbent v3 dominates failed challengers")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--reference-v3-oof", default=DEFAULT_V3_OOF_SPEC)
    parser.add_argument(
        "--challenger",
        action="append",
        default=[],
        help="Optional challenger override: <name>|<path>|<prediction_column>|<family>",
    )
    parser.add_argument("--label", default="v3_dominance")
    parser.add_argument("--out-dir", default="artifacts/reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    challengers = tuple(_parse_challenger(raw) for raw in args.challenger) if args.challenger else DEFAULT_CHALLENGERS
    result = run_v3_dominance_diagnostic(
        train_csv_path=args.train_csv,
        reference_oof_spec=args.reference_v3_oof,
        challengers=challengers,
        out_dir=args.out_dir,
        label=args.label,
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
