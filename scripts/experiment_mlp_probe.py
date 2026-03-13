#!/usr/bin/env python3
"""Run a shallow tabular MLP probe and compare it with the incumbent blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.diagnostics import load_merged_oof_matrix, parse_feature_blocks_arg, parse_oof_input_spec
from churn_baseline.mlp_probe import MLPProbeParams, run_mlp_probe_cv
from churn_baseline.specialist import build_reference_prediction


def _parse_alpha_grid(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"alpha values must be in [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("alpha grid must contain at least one value")
    return values


def _parse_hidden_dims(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("hidden dims must contain at least one integer")
    return tuple(int(part) for part in parts)


def _load_weights(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Could not find weights in {path}")
    return {str(name): float(value) for name, value in weights.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tabular embedding MLP probe smoke test")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument(
        "--feature-blocks",
        default="H,R,S,V",
        help="Comma-separated feature blocks for the MLP probe.",
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-dims", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-embedding-dim", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--model-path",
        default="artifacts/models/mlp_probe_smoke.pt",
        help="Output torch bundle path for the full-train MLP probe.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/mlp_probe_smoke_metrics.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/mlp_probe_smoke_oof.csv",
        help="Output OOF CSV path.",
    )
    parser.add_argument(
        "--reference-weights-json",
        default="",
        help="Optional weights JSON for incumbent reference blend.",
    )
    parser.add_argument(
        "--oof",
        action="append",
        default=[],
        help="OOF input spec: <name>=<path>[#<prediction_column>]. Required when using --reference-weights-json.",
    )
    parser.add_argument(
        "--alpha-grid",
        default="0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.50",
        help="Blend alphas for probe-vs-incumbent scan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_pred = None
    if args.reference_weights_json:
        if not args.oof:
            raise ValueError("--oof is required when --reference-weights-json is provided.")
        specs = [parse_oof_input_spec(raw) for raw in args.oof]
        merged_oof, _ = load_merged_oof_matrix(specs)
        reference_pred = build_reference_prediction(merged_oof, _load_weights(args.reference_weights_json))

    params = MLPProbeParams(
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        max_embedding_dim=int(args.max_embedding_dim),
        device=str(args.device),
    )
    metrics = run_mlp_probe_cv(
        train_csv_path=args.train_csv,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
        feature_blocks=parse_feature_blocks_arg(args.feature_blocks),
        folds=args.folds,
        random_state=args.random_state,
        params=params,
        reference_pred=reference_pred,
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
