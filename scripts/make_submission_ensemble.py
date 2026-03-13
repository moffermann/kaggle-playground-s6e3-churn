#!/usr/bin/env python3
"""Generate submission by averaging multiple trained models."""

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.pipeline import make_submission_ensemble


def _parse_model_paths(raw: str) -> list[str]:
    paths = [token.strip() for token in raw.split(",") if token.strip()]
    if not paths:
        raise ValueError("At least one model path is required")
    return paths


def _model_paths_from_metrics(metrics_path: str | Path) -> list[str]:
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    model_paths = payload.get("model_paths", [])
    if not model_paths:
        raise ValueError(f"No model_paths found in metrics file: {metrics_path}")
    return [str(path) for path in model_paths]


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ensemble submission from multiple models")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--output-csv",
        default="artifacts/submissions/playground-series-s6e3-ensemble.csv",
        help="Output submission CSV",
    )
    parser.add_argument(
        "--model-paths",
        default="",
        help="Comma-separated model paths. If empty, reads --metrics-path",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/train_cv_multiseed_metrics.json",
        help="Metrics JSON with model_paths key",
    )
    parser.add_argument(
        "--feature-blocks",
        default="none",
        help="Optional feature blocks (A,B,C,F,G,T,H,R,S,V,O,P) or none.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_blocks = _parse_feature_blocks(args.feature_blocks)
    if args.model_paths.strip():
        model_paths = _parse_model_paths(args.model_paths)
    else:
        model_paths = _model_paths_from_metrics(args.metrics_path)

    result = make_submission_ensemble(
        model_paths=model_paths,
        test_csv_path=args.test_csv,
        output_csv_path=args.output_csv,
        feature_blocks=feature_blocks,
        train_csv_path=args.train_csv,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
