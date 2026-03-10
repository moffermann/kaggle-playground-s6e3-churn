#!/usr/bin/env python3
"""Build teacher components if needed, then materialize a residual submission in one command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.diagnostics import parse_feature_blocks_arg
from churn_baseline.specialist import (
    build_teacher_reference_component_frame,
    make_residual_reranker_chain_submission,
)


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


def _load_required_json_key(path: str | Path, key: str) -> object:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    value = payload.get(key)
    if value is None:
        raise ValueError(f"Could not find key '{key}' in {path}")
    return value


def _resolve_component_output_path(raw: str, output_csv: str, suffix: str) -> Path:
    if raw.strip():
        return Path(raw)
    output_path = Path(output_csv)
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def _resolve_component_report_path(raw: str, report_json: str, suffix: str) -> Path:
    if raw.strip():
        return Path(raw)
    report_path = Path(report_json)
    return report_path.with_name(f"{report_path.stem}{suffix}{report_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build teacher components if needed and materialize a residual submission"
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
        help="Final residual report JSON path.",
    )
    parser.add_argument(
        "--reference-mode",
        default="previous",
        choices=["previous", "base"],
        help="Use the previous step output or the base reference submission as the step teacher.",
    )
    parser.add_argument(
        "--teacher-aware",
        action="store_true",
        help="Build the teacher component frame and pass it to teacher-aware rerankers.",
    )
    parser.add_argument(
        "--reference-component-csv",
        default="",
        help="Optional output/input CSV with id + pred_* columns for teacher-aware rerankers.",
    )
    parser.add_argument(
        "--reference-component-report-json",
        default="",
        help="Optional JSON path for the teacher component build report.",
    )
    parser.add_argument(
        "--cb-metrics-json",
        default="artifacts/reports/train_cv_multiseed_gate_s5_hiiter_metrics.json",
        help="Metrics JSON with cb model_paths.",
    )
    parser.add_argument(
        "--cb-feature-blocks",
        default="none",
        help="Feature blocks used by the cb ensemble models.",
    )
    parser.add_argument(
        "--r-model-path",
        default="artifacts/models/fe_blockR_hiiter.cbm",
        help="CatBoost model path for the R component.",
    )
    parser.add_argument(
        "--r-feature-blocks",
        default="R",
        help="Feature blocks used by the R component model.",
    )
    parser.add_argument(
        "--cbxgblgb-submission",
        default="artifacts/submissions/playground-series-s6e3.csv",
        help="Validated submission for the cb+xgb+lgb blend.",
    )
    parser.add_argument(
        "--cbxgblgb-weights-json",
        default="artifacts/reports/submission_candidate_cb5_xgb3_lgb_weights.json",
        help="Weights JSON for the cb+xgb+lgb blend.",
    )
    parser.add_argument(
        "--cbr-submission",
        default="artifacts/submissions/playground-series-s6e3-rblend.csv",
        help="Validated submission for the cb+xgb+lgb+r blend.",
    )
    parser.add_argument(
        "--cbr-weights-json",
        default="artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_weights.json",
        help="Weights JSON for the cb+xgb+lgb+r blend.",
    )
    parser.add_argument(
        "--cbrv-submission",
        default="artifacts/submissions/playground-series-s6e3-rvblend.csv",
        help="Validated submission for the cb+xgb+lgb+r+rv blend.",
    )
    parser.add_argument(
        "--cbrv-weights-json",
        default="artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json",
        help="Weights JSON for the cb+xgb+lgb+r+rv blend.",
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

    reference_component_frame = None
    teacher_component_csv_path = None
    teacher_component_report_path = None

    if args.teacher_aware:
        if args.reference_mode != "base":
            raise ValueError("--teacher-aware requires --reference-mode base for train/inference parity")

        teacher_component_csv_path = _resolve_component_output_path(
            args.reference_component_csv,
            args.output_csv,
            "_teacher_components",
        )
        teacher_component_report_path = _resolve_component_report_path(
            args.reference_component_report_json,
            args.report_json,
            "_teacher_components",
        )

        component_frame, component_report = build_teacher_reference_component_frame(
            test_csv_path=args.test_csv,
            train_csv_path=args.train_csv,
            cb_model_paths=_load_required_json_key(args.cb_metrics_json, "model_paths"),
            cb_feature_blocks=parse_feature_blocks_arg(args.cb_feature_blocks),
            r_model_path=args.r_model_path,
            r_feature_blocks=parse_feature_blocks_arg(args.r_feature_blocks),
            cbxgblgb_submission_path=args.cbxgblgb_submission,
            cbxgblgb_weights=_load_required_json_key(args.cbxgblgb_weights_json, "weights"),
            cbr_submission_path=args.cbr_submission,
            cbr_weights=_load_required_json_key(args.cbr_weights_json, "weights"),
            cbrv_submission_path=args.cbrv_submission,
            cbrv_weights=_load_required_json_key(args.cbrv_weights_json, "weights"),
        )
        teacher_component_csv_path.parent.mkdir(parents=True, exist_ok=True)
        component_frame.to_csv(teacher_component_csv_path, index=False)
        teacher_component_report_path.parent.mkdir(parents=True, exist_ok=True)
        teacher_component_report = {
            **component_report,
            "output_csv_path": str(teacher_component_csv_path),
            "report_json_path": str(teacher_component_report_path),
        }
        teacher_component_report_path.write_text(json.dumps(teacher_component_report, indent=2), encoding="utf-8")
        reference_component_frame = component_frame
    elif args.reference_component_csv.strip():
        reference_component_frame = pd.read_csv(args.reference_component_csv)

    run_config = {
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "reference_submission": args.reference_submission,
        "reference_mode": args.reference_mode,
        "teacher_aware": bool(args.teacher_aware),
        "reference_component_csv": str(teacher_component_csv_path or args.reference_component_csv or ""),
        "output_csv": args.output_csv,
        "report_json": args.report_json,
        "steps": steps,
    }
    print(json.dumps({"run_config": run_config}, indent=2))

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
