#!/usr/bin/env python3
"""Run counterfactual teacher-sensitivity smoke directly against incumbent v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.counterfactual_sensitivity import (
    DEFAULT_COMPONENT_WEIGHTS_JSON,
    SUPPORTED_COUNTERFACTUALS,
    SUPPORTED_SIGNALS,
    run_counterfactual_sensitivity_smoke,
)
from churn_baseline.uncertainty_band import DEFAULT_V3_OOF_SPEC
from churn_baseline.validation_protocol import DOMINANT_MACROFAMILY, SUPPORTED_STAGES, evaluate_validation_protocol


def _parse_csv_tokens(raw: str, *, allowed: tuple[str, ...], argument_name: str) -> list[str]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{argument_name} must contain at least one value")
    invalid = [token for token in tokens if token not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported values for {argument_name}: {invalid}. Supported: {list(allowed)}"
        )
    return tokens


def _parse_alpha_grid(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("alpha grid must contain at least one value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run counterfactual teacher sensitivity directly against v3")
    parser.add_argument("--stage", choices=SUPPORTED_STAGES, default="smoke")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument(
        "--reference-v3-oof",
        default=DEFAULT_V3_OOF_SPEC,
        help="Direct OOF spec for incumbent v3 (<path>[#prediction_column]).",
    )
    parser.add_argument(
        "--component-weights-json",
        default=DEFAULT_COMPONENT_WEIGHTS_JSON,
        help="Weights JSON used to select and normalize the teacher component subset.",
    )
    parser.add_argument(
        "--target-family-level",
        choices=["segment3", "segment5"],
        default="segment3",
    )
    parser.add_argument(
        "--target-family-value",
        default=DOMINANT_MACROFAMILY,
        help="Family key used to localize the counterfactual correction.",
    )
    parser.add_argument(
        "--reference-band-half-width",
        type=float,
        default=None,
        help="Optional extra filter: only rows with abs(v3 - 0.5) <= this width are corrected.",
    )
    parser.add_argument(
        "--counterfactuals",
        default="auto_payment,paperless_off,contract_upgrade,stability_bundle",
        help="Comma-separated counterfactual scenarios to evaluate.",
    )
    parser.add_argument(
        "--signal-names",
        default="stable_bundle_drop,mean_positive_drop,max_positive_drop",
        help="Comma-separated aggregate sensitivity signals to scan.",
    )
    parser.add_argument(
        "--alpha-grid",
        default="-0.20,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.30",
        help="Comma-separated additive scales applied over the selected signal.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/counterfactual_teacher_smoke_metrics.json",
    )
    parser.add_argument(
        "--oof-path",
        default="artifacts/reports/counterfactual_teacher_smoke_analysis_oof.csv",
    )
    parser.add_argument(
        "--gate-path",
        default="artifacts/reports/validation_protocol_counterfactual_teacher_smoke_vs_v3.json",
        help="Validation reset report written after the smoke run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counterfactuals = _parse_csv_tokens(
        args.counterfactuals,
        allowed=SUPPORTED_COUNTERFACTUALS,
        argument_name="--counterfactuals",
    )
    signal_names = _parse_csv_tokens(
        args.signal_names,
        allowed=SUPPORTED_SIGNALS,
        argument_name="--signal-names",
    )
    alpha_grid = _parse_alpha_grid(args.alpha_grid)

    metrics = run_counterfactual_sensitivity_smoke(
        train_csv_path=args.train_csv,
        reference_v3_oof=args.reference_v3_oof,
        component_weights_json=args.component_weights_json,
        target_family_level=args.target_family_level,
        target_family_value=args.target_family_value,
        reference_band_half_width=args.reference_band_half_width,
        alpha_grid=alpha_grid,
        counterfactuals=counterfactuals,
        signal_names=signal_names,
        metrics_path=args.metrics_path,
        oof_path=args.oof_path,
    )

    gate = evaluate_validation_protocol(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        stage=args.stage,
        analysis_oof_path=args.oof_path,
        target_family_level=args.target_family_level,
        target_family_value=args.target_family_value,
        dominant_family_value=DOMINANT_MACROFAMILY,
        candidate_metrics_json=args.metrics_path,
        out_json_path=args.gate_path,
    )

    summary = {
        "metrics_path": str(Path(args.metrics_path)),
        "oof_path": str(Path(args.oof_path)),
        "gate_path": str(Path(args.gate_path)),
        "metrics": metrics,
        "gate": gate,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
