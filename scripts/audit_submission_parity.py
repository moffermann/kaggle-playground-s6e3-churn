#!/usr/bin/env python3
"""Audit train/inference parity for submission generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.config import ID_COLUMN, TARGET_COLUMN
from churn_baseline.data import infer_categorical_columns, load_csv, prepare_test_features, prepare_train_features
from churn_baseline.diagnostics import (
    FAIL_STATUS,
    PASS_STATUS,
    WARN_STATUS,
    make_check,
    parse_feature_blocks_arg,
    summarize_checks,
    utc_now_iso,
)


def _cap(items: list[str], limit: int = 25) -> list[str]:
    if len(items) <= limit:
        return items
    return [*items[:limit], f"... (+{len(items) - limit} more)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit submission parity and artifact consistency")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--feature-blocks",
        default="none",
        help="Optional feature blocks (A,B,C,O,P) or none.",
    )
    parser.add_argument(
        "--metrics-path",
        default="artifacts/reports/train_cv_multiseed_metrics.json",
        help="Metrics JSON used to produce submission.",
    )
    parser.add_argument(
        "--submission-csv",
        default="",
        help="Optional submission CSV to validate.",
    )
    parser.add_argument(
        "--reference-submission-csv",
        default="",
        help="Optional reference submission CSV for prediction-delta checks.",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/diagnostic_submission_parity.json",
        help="Output audit report JSON path.",
    )
    return parser.parse_args()


def _load_metrics_if_present(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in metrics file: {metrics_path}")
    return payload


def _extract_model_paths(metrics: dict) -> list[str]:
    paths: list[str] = []
    if "model_path" in metrics and isinstance(metrics["model_path"], str):
        paths.append(metrics["model_path"])
    model_paths = metrics.get("model_paths")
    if isinstance(model_paths, list):
        for item in model_paths:
            if isinstance(item, str):
                paths.append(item)
    deduped = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def main() -> int:
    args = parse_args()
    feature_blocks = parse_feature_blocks_arg(args.feature_blocks)

    train_df = load_csv(args.train_csv)
    test_df = load_csv(args.test_csv)
    x_train, _ = prepare_train_features(train_df, drop_id=True, feature_blocks=feature_blocks)
    x_test = prepare_test_features(test_df, drop_id=True, feature_blocks=feature_blocks)

    checks = []

    train_raw_feature_cols = [col for col in train_df.columns if col != TARGET_COLUMN]
    test_raw_cols = list(test_df.columns)
    missing_raw_in_test = sorted(set(train_raw_feature_cols) - set(test_raw_cols))
    extra_raw_in_test = sorted(set(test_raw_cols) - set(train_raw_feature_cols))
    raw_status = PASS_STATUS if not missing_raw_in_test and not extra_raw_in_test else FAIL_STATUS
    checks.append(
        make_check(
            "raw_columns_train_vs_test",
            raw_status,
            {
                "train_feature_column_count": len(train_raw_feature_cols),
                "test_column_count": len(test_raw_cols),
                "missing_in_test": _cap(missing_raw_in_test),
                "extra_in_test": _cap(extra_raw_in_test),
            },
        )
    )

    train_cols = list(x_train.columns)
    test_cols = list(x_test.columns)
    missing_engineered_in_test = sorted(set(train_cols) - set(test_cols))
    extra_engineered_in_test = sorted(set(test_cols) - set(train_cols))
    engineered_set_status = (
        PASS_STATUS if not missing_engineered_in_test and not extra_engineered_in_test else FAIL_STATUS
    )
    checks.append(
        make_check(
            "engineered_columns_set_match",
            engineered_set_status,
            {
                "train_feature_count": len(train_cols),
                "test_feature_count": len(test_cols),
                "missing_in_test": _cap(missing_engineered_in_test),
                "extra_in_test": _cap(extra_engineered_in_test),
            },
        )
    )

    order_match = train_cols == test_cols
    checks.append(
        make_check(
            "engineered_columns_order_match",
            PASS_STATUS if order_match else FAIL_STATUS,
            {
                "order_match": bool(order_match),
                "first_train_columns": train_cols[:10],
                "first_test_columns": test_cols[:10],
            },
        )
    )

    common_cols = [column for column in train_cols if column in x_test.columns]
    dtype_mismatches = []
    for column in common_cols:
        train_dtype = str(x_train[column].dtype)
        test_dtype = str(x_test[column].dtype)
        if train_dtype != test_dtype:
            dtype_mismatches.append(
                {"column": column, "train_dtype": train_dtype, "test_dtype": test_dtype}
            )
    dtype_status = PASS_STATUS if not dtype_mismatches else WARN_STATUS
    checks.append(
        make_check(
            "engineered_dtype_match",
            dtype_status,
            {
                "mismatch_count": len(dtype_mismatches),
                "examples": dtype_mismatches[:20],
            },
        )
    )

    train_cat = set(infer_categorical_columns(x_train))
    test_cat = set(infer_categorical_columns(x_test))
    missing_cat_in_test = sorted(train_cat - test_cat)
    extra_cat_in_test = sorted(test_cat - train_cat)
    cat_status = PASS_STATUS if not missing_cat_in_test and not extra_cat_in_test else WARN_STATUS
    checks.append(
        make_check(
            "categorical_columns_match",
            cat_status,
            {
                "train_categorical_count": len(train_cat),
                "test_categorical_count": len(test_cat),
                "missing_in_test": _cap(missing_cat_in_test),
                "extra_in_test": _cap(extra_cat_in_test),
            },
        )
    )

    metrics_path = Path(args.metrics_path)
    metrics_payload = _load_metrics_if_present(metrics_path)
    if metrics_payload is None:
        checks.append(
            make_check(
                "metrics_file_presence",
                WARN_STATUS,
                {
                    "metrics_path": str(metrics_path),
                    "exists": False,
                },
            )
        )
    else:
        checks.append(
            make_check(
                "metrics_file_presence",
                PASS_STATUS,
                {
                    "metrics_path": str(metrics_path),
                    "exists": True,
                },
            )
        )

        metric_blocks = metrics_payload.get("feature_blocks")
        block_status = PASS_STATUS
        if metric_blocks is None:
            if feature_blocks:
                block_status = WARN_STATUS
        elif isinstance(metric_blocks, list):
            if list(metric_blocks) != feature_blocks:
                block_status = FAIL_STATUS
        else:
            block_status = WARN_STATUS
        checks.append(
            make_check(
                "metrics_feature_blocks_match",
                block_status,
                {
                    "requested_feature_blocks": feature_blocks,
                    "metrics_feature_blocks": metric_blocks,
                },
            )
        )

        metric_feature_count = metrics_payload.get("feature_count")
        feature_count_status = PASS_STATUS
        if isinstance(metric_feature_count, int):
            if int(metric_feature_count) != int(x_train.shape[1]):
                feature_count_status = FAIL_STATUS
        else:
            feature_count_status = WARN_STATUS
        checks.append(
            make_check(
                "metrics_feature_count_match",
                feature_count_status,
                {
                    "prepared_feature_count": int(x_train.shape[1]),
                    "metrics_feature_count": metric_feature_count,
                },
            )
        )

        model_paths = _extract_model_paths(metrics_payload)
        missing_model_paths = [path for path in model_paths if not Path(path).exists()]
        model_status = PASS_STATUS if not missing_model_paths else FAIL_STATUS
        checks.append(
            make_check(
                "metrics_model_paths_exist",
                model_status,
                {
                    "model_path_count": len(model_paths),
                    "missing_model_paths": _cap(missing_model_paths),
                },
            )
        )

    submission_path = Path(args.submission_csv) if args.submission_csv else None
    if submission_path is None:
        checks.append(
            make_check(
                "submission_csv_provided",
                WARN_STATUS,
                {
                    "provided": False,
                },
            )
        )
    else:
        if not submission_path.exists():
            checks.append(
                make_check(
                    "submission_csv_exists",
                    FAIL_STATUS,
                    {
                        "submission_csv": str(submission_path),
                        "exists": False,
                    },
                )
            )
        else:
            submission_df = load_csv(submission_path)
            required = {ID_COLUMN, TARGET_COLUMN}
            missing_submission_cols = sorted(required - set(submission_df.columns))
            checks.append(
                make_check(
                    "submission_required_columns",
                    PASS_STATUS if not missing_submission_cols else FAIL_STATUS,
                    {
                        "columns": list(submission_df.columns),
                        "missing_required_columns": missing_submission_cols,
                    },
                )
            )

            row_match = len(submission_df) == len(test_df)
            checks.append(
                make_check(
                    "submission_row_count_match_test",
                    PASS_STATUS if row_match else FAIL_STATUS,
                    {
                        "submission_rows": int(len(submission_df)),
                        "test_rows": int(len(test_df)),
                    },
                )
            )

            if ID_COLUMN in submission_df.columns and ID_COLUMN in test_df.columns:
                id_match = submission_df[ID_COLUMN].equals(test_df[ID_COLUMN])
                checks.append(
                    make_check(
                        "submission_id_order_match_test",
                        PASS_STATUS if id_match else FAIL_STATUS,
                        {
                            "id_order_match": bool(id_match),
                        },
                    )
                )

            if TARGET_COLUMN in submission_df.columns:
                pred_series = submission_df[TARGET_COLUMN]
                pred_min = float(pred_series.min())
                pred_max = float(pred_series.max())
                in_range = pred_min >= 0.0 and pred_max <= 1.0
                checks.append(
                    make_check(
                        "submission_prediction_range",
                        PASS_STATUS if in_range else WARN_STATUS,
                        {
                            "prediction_min": pred_min,
                            "prediction_max": pred_max,
                            "expected_range": "[0,1]",
                        },
                    )
                )

            reference_path = Path(args.reference_submission_csv) if args.reference_submission_csv else None
            if reference_path is None:
                checks.append(
                    make_check(
                        "reference_submission_provided",
                        WARN_STATUS,
                        {"provided": False},
                    )
                )
            elif not reference_path.exists():
                checks.append(
                    make_check(
                        "reference_submission_exists",
                        FAIL_STATUS,
                        {"reference_submission_csv": str(reference_path), "exists": False},
                    )
                )
            else:
                reference_df = load_csv(reference_path)
                if ID_COLUMN in reference_df.columns and TARGET_COLUMN in reference_df.columns:
                    aligned = (
                        len(reference_df) == len(submission_df)
                        and submission_df[ID_COLUMN].equals(reference_df[ID_COLUMN])
                    )
                    if not aligned:
                        checks.append(
                            make_check(
                                "reference_submission_alignment",
                                FAIL_STATUS,
                                {
                                    "aligned_on_id_and_rows": False,
                                    "reference_rows": int(len(reference_df)),
                                    "submission_rows": int(len(submission_df)),
                                },
                            )
                        )
                    else:
                        delta = (submission_df[TARGET_COLUMN] - reference_df[TARGET_COLUMN]).astype("float64")
                        corr = float(submission_df[TARGET_COLUMN].corr(reference_df[TARGET_COLUMN]))
                        mae = float(delta.abs().mean())
                        max_abs_delta = float(delta.abs().max())
                        status = PASS_STATUS if corr >= 0.95 else WARN_STATUS
                        checks.append(
                            make_check(
                                "reference_submission_delta",
                                status,
                                {
                                    "reference_submission_csv": str(reference_path),
                                    "pearson_correlation": corr,
                                    "mean_absolute_delta": mae,
                                    "max_absolute_delta": max_abs_delta,
                                },
                            )
                        )
                else:
                    checks.append(
                        make_check(
                            "reference_submission_required_columns",
                            FAIL_STATUS,
                            {
                                "reference_submission_csv": str(reference_path),
                                "columns": list(reference_df.columns),
                            },
                        )
                    )

    summary = summarize_checks(checks)
    report = {
        "generated_at_utc": utc_now_iso(),
        "inputs": {
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "feature_blocks": feature_blocks,
            "metrics_path": args.metrics_path,
            "submission_csv": args.submission_csv or None,
            "reference_submission_csv": args.reference_submission_csv or None,
        },
        "prepared_features": {
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "feature_count": int(x_train.shape[1]),
            "categorical_column_count": int(len(train_cat)),
        },
        **summary,
        "checks": checks,
    }
    write_json(args.out_json, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
