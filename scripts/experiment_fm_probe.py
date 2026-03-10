#!/usr/bin/env python3
"""Run River FM/FFM smoke tests and compare them with the incumbent blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.diagnostics import load_merged_oof_matrix, parse_feature_blocks_arg, parse_oof_input_spec
from churn_baseline.fm_probe import list_fm_probe_families, run_fm_probe_cv
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


def _load_weights(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Could not find weights in {path}")
    return {str(name): float(value) for name, value in weights.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run River FM/FFM probe smoke test")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument(
        "--feature-blocks",
        default="H,R,S,V",
        help="Comma-separated feature blocks for the FM probe.",
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-family",
        default="ffm",
        choices=list(list_fm_probe_families()),
        help="Factorization model family to evaluate.",
    )
    parser.add_argument("--n-factors", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--weight-lr", type=float, default=0.05)
    parser.add_argument("--latent-lr", type=float, default=0.03)
    parser.add_argument("--l2-weight", type=float, default=1e-6)
    parser.add_argument("--l2-latent", type=float, default=1e-6)
    parser.add_argument(
        "--no-sample-normalization",
        action="store_true",
        help="Disable River sample normalization.",
    )
    parser.add_argument(
        "--model-path",
        default="artifacts/models/fm_probe_smoke.pkl",
        help="Output pickle path for the full-train FM probe.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/fm_probe_smoke_metrics.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/fm_probe_smoke_oof.csv",
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

    metrics = run_fm_probe_cv(
        train_csv_path=args.train_csv,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
        feature_blocks=parse_feature_blocks_arg(args.feature_blocks),
        folds=args.folds,
        random_state=args.random_state,
        model_family=args.model_family,
        n_factors=args.n_factors,
        epochs=args.epochs,
        weight_lr=args.weight_lr,
        latent_lr=args.latent_lr,
        l2_weight=args.l2_weight,
        l2_latent=args.l2_latent,
        sample_normalization=not args.no_sample_normalization,
        reference_pred=reference_pred,
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
