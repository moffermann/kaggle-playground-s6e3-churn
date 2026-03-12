#!/usr/bin/env python3
"""Run the minimal GraphSAGE probe experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.gnn_probe import GNNProbeParams, train_gnn_probe_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal transductive GraphSAGE smoke")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--graph-numeric-multiplier", type=float, default=3.0)
    parser.add_argument("--metrics-path", default="artifacts/reports/gnn_probe_smoke_metrics.json")
    parser.add_argument("--oof-path", default="artifacts/reports/gnn_probe_smoke_oof.csv")
    parser.add_argument("--test-pred-path", default="")
    parser.add_argument("--reference-v3-oof", default="")
    parser.add_argument("--analysis-oof-path", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = GNNProbeParams(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        k_neighbors=args.k_neighbors,
        graph_numeric_multiplier=args.graph_numeric_multiplier,
        random_state=args.random_state,
    )
    metrics = train_gnn_probe_cv(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
        test_pred_path=args.test_pred_path.strip() or None,
        analysis_oof_path=args.analysis_oof_path.strip() or None,
        reference_v3_oof_path=args.reference_v3_oof.strip() or None,
        params=params,
        folds=args.folds,
        random_state=args.random_state,
        device=args.device,
    )
    Path(args.metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
