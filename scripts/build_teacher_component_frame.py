#!/usr/bin/env python3
"""Build a teacher reference component frame for teacher-aware residual inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.diagnostics import parse_feature_blocks_arg
from churn_baseline.specialist import build_teacher_reference_component_frame


def _load_weights(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Could not find weights in {path}")
    return {str(name): float(value) for name, value in weights.items()}


def _load_model_paths(metrics_path: str | Path) -> list[str]:
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    model_paths = payload.get("model_paths")
    if not isinstance(model_paths, list) or not model_paths:
        raise ValueError(f"Could not find model_paths in {metrics_path}")
    return [str(path) for path in model_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build teacher reference component frame from incumbent blend artifacts"
    )
    parser.add_argument("--test-csv", default="data/raw/test.csv")
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
        "--output-csv",
        default="artifacts/reports/reference_component_frame_rvblend_test.csv",
        help="Output CSV with id + pred_* columns.",
    )
    parser.add_argument(
        "--report-json",
        default="artifacts/reports/reference_component_frame_rvblend_test_report.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    component_frame, report = build_teacher_reference_component_frame(
        test_csv_path=args.test_csv,
        cb_model_paths=_load_model_paths(args.cb_metrics_json),
        cb_feature_blocks=parse_feature_blocks_arg(args.cb_feature_blocks),
        r_model_path=args.r_model_path,
        r_feature_blocks=parse_feature_blocks_arg(args.r_feature_blocks),
        cbxgblgb_submission_path=args.cbxgblgb_submission,
        cbxgblgb_weights=_load_weights(args.cbxgblgb_weights_json),
        cbr_submission_path=args.cbr_submission,
        cbr_weights=_load_weights(args.cbr_weights_json),
        cbrv_submission_path=args.cbrv_submission,
        cbrv_weights=_load_weights(args.cbrv_weights_json),
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    component_frame.to_csv(output_path, index=False)

    report = {
        **report,
        "output_csv_path": str(output_path),
        "report_json_path": str(args.report_json),
    }
    Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
