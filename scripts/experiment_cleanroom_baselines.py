#!/usr/bin/env python3
"""Run the minimal clean-room baseline smoke suite directly against v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.cleanroom_baseline import run_cleanroom_baseline_suite
from churn_baseline.config import CatBoostHyperParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal clean-room baseline smoke suite")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--v3-oof", default="artifacts/reports/validation_protocol_v3_chain_oof.csv#candidate_pred")
    parser.add_argument("--out-dir", default="artifacts/reports")
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--out-summary-json", default="artifacts/reports/cleanroom_baseline_suite_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = CatBoostHyperParams(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_state,
    )
    summary = run_cleanroom_baseline_suite(
        train_csv_path=args.train_csv,
        v3_oof_spec=args.v3_oof,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        params=params,
        folds=args.folds,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
    )
    write_json(Path(args.out_summary_json), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
