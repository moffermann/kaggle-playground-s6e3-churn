#!/usr/bin/env python3
"""Analyze OOF error by customer class/cohort for incumbent and challenger blends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json
from churn_baseline.config import ID_COLUMN, TARGET_COLUMN
from churn_baseline.data import load_csv
from churn_baseline.diagnostics import (
    OOF_TARGET_COLUMN,
    load_merged_oof_matrix,
    parse_oof_input_spec,
    utc_now_iso,
)
from churn_baseline.evaluation import binary_auc
from churn_baseline.feature_engineering import apply_feature_engineering


DEFAULT_OOF_SPECS = (
    "cb=artifacts/reports/train_cv_multiseed_gate_s5_hiiter_oof.csv#oof_ensemble",
    "xgb=artifacts/reports/train_xgboost_cv_multiseed_hiiter_oof.csv#oof_ensemble",
    "lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred",
    "r=artifacts/reports/fe_blockR_hiiter_oof.csv#oof_pred",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze OOF error by class/cohort")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument(
        "--oof",
        action="append",
        default=[],
        help="OOF input spec: <name>=<path>[#<prediction_column>]. Defaults to cb/xgb/lgb/r.",
    )
    parser.add_argument(
        "--reference-weights-json",
        default="artifacts/reports/submission_candidate_cb5_xgb3_lgb_weights.json",
        help="Weights JSON for incumbent/reference blend.",
    )
    parser.add_argument(
        "--challenger-weights-json",
        default="artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_weights.json",
        help="Weights JSON for challenger/current-best blend.",
    )
    parser.add_argument(
        "--hard-loss-quantile",
        type=float,
        default=0.99,
        help="Quantile of per-row logloss used to define hard cases.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Rows per summary section.")
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/diagnostic_error_by_class_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--out-groups-csv",
        default="artifacts/reports/diagnostic_error_by_class_groups.csv",
        help="Output combined group metrics CSV path.",
    )
    parser.add_argument(
        "--out-hard-cases-csv",
        default="artifacts/reports/diagnostic_error_by_class_hard_cases.csv",
        help="Output hard-case hotspot CSV path.",
    )
    return parser.parse_args()


def _load_weights(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Could not find weights in {path}")
    return {str(name): float(value) for name, value in weights.items()}


def _safe_auc(y_true: pd.Series, pred: pd.Series) -> float | None:
    if y_true.nunique(dropna=False) < 2:
        return None
    return float(binary_auc(y_true.astype(int), pred.astype("float64")))


def _logloss_vector(y_true: pd.Series, pred: pd.Series, epsilon: float = 1e-6) -> pd.Series:
    clipped = pred.astype("float64").clip(lower=epsilon, upper=1.0 - epsilon)
    y = y_true.astype("float64")
    return -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))


def _build_analysis_frame(train_csv: str, merged_oof: pd.DataFrame) -> pd.DataFrame:
    train_df = load_csv(train_csv)
    if ID_COLUMN not in train_df.columns or TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"{train_csv} must contain {ID_COLUMN} and {TARGET_COLUMN}")

    base = train_df.drop(columns=[TARGET_COLUMN]).copy()
    engineered = apply_feature_engineering(base.drop(columns=[ID_COLUMN]), feature_blocks=["A", "R"])
    frame = pd.concat([base[[ID_COLUMN]], engineered], axis=1)

    frame["segment3"] = (
        frame["PaymentMethod"].astype(str)
        + "__"
        + frame["Contract"].astype(str)
        + "__"
        + frame["InternetService"].astype(str)
    )
    frame["segment5"] = (
        frame["PaymentMethod"].astype(str)
        + "__"
        + frame["Contract"].astype(str)
        + "__"
        + frame["InternetService"].astype(str)
        + "__"
        + frame["PaperlessBilling"].astype(str)
        + "__"
        + frame["tenure_bin"].astype(str)
    )
    segment5_support = frame.groupby("segment5").size()
    frame["segment5_support"] = frame["segment5"].map(segment5_support).astype("int32")
    frame["segment5_support_bucket"] = pd.cut(
        frame["segment5_support"],
        bins=[0, 49, 99, 249, 499, np.inf],
        labels=["lt50", "50_99", "100_249", "250_499", "500_plus"],
        right=True,
    ).astype(str)

    merged = frame.merge(merged_oof, how="inner", on=ID_COLUMN, validate="one_to_one")
    if merged.empty:
        raise ValueError("Merged analysis frame is empty.")
    return merged


def _compute_overall_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    y_true = frame[OOF_TARGET_COLUMN].astype(int)
    pred = frame[pred_col].astype("float64")
    losses = _logloss_vector(y_true, pred)
    squared_error = (pred - y_true) ** 2
    abs_error = (pred - y_true).abs()
    positive_rate = float(y_true.mean())
    pred_mean = float(pred.mean())
    return {
        "auc": float(binary_auc(y_true, pred)),
        "logloss": float(losses.mean()),
        "brier": float(squared_error.mean()),
        "mae": float(abs_error.mean()),
        "positive_rate": positive_rate,
        "pred_mean": pred_mean,
        "calibration_gap": float(pred_mean - positive_rate),
        "pred_std": float(pred.std(ddof=0)),
    }


def _group_metrics(
    frame: pd.DataFrame,
    *,
    family: str,
    column: str,
    min_rows: int,
    challenger_col: str,
    reference_col: str,
    challenger_loss_col: str,
    reference_loss_col: str,
    challenger_hard_col: str,
    reference_hard_col: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_rows = float(len(frame))

    for group_value, part in frame.groupby(column, dropna=False):
        if len(part) < min_rows:
            continue

        y_true = part[OOF_TARGET_COLUMN].astype(int)
        challenger_pred = part[challenger_col].astype("float64")
        reference_pred = part[reference_col].astype("float64")

        challenger_mean = float(challenger_pred.mean())
        reference_mean = float(reference_pred.mean())
        positive_rate = float(y_true.mean())
        challenger_auc = _safe_auc(y_true, challenger_pred)
        reference_auc = _safe_auc(y_true, reference_pred)

        challenger_logloss = float(part[challenger_loss_col].mean())
        reference_logloss = float(part[reference_loss_col].mean())
        challenger_brier = float(((challenger_pred - y_true) ** 2).mean())
        reference_brier = float(((reference_pred - y_true) ** 2).mean())
        challenger_mae = float((challenger_pred - y_true).abs().mean())
        reference_mae = float((reference_pred - y_true).abs().mean())

        row = {
            "family": family,
            "column": column,
            "group_value": str(group_value),
            "rows": int(len(part)),
            "positives": int(y_true.sum()),
            "positive_rate": positive_rate,
            "challenger_auc": challenger_auc,
            "reference_auc": reference_auc,
            "delta_auc": (
                None if challenger_auc is None or reference_auc is None else float(challenger_auc - reference_auc)
            ),
            "challenger_logloss": challenger_logloss,
            "reference_logloss": reference_logloss,
            "delta_logloss": float(challenger_logloss - reference_logloss),
            "logloss_delta_contribution": float(
                (challenger_logloss - reference_logloss) * float(len(part)) / total_rows
            ),
            "challenger_brier": challenger_brier,
            "reference_brier": reference_brier,
            "delta_brier": float(challenger_brier - reference_brier),
            "challenger_mae": challenger_mae,
            "reference_mae": reference_mae,
            "delta_mae": float(challenger_mae - reference_mae),
            "challenger_pred_mean": challenger_mean,
            "reference_pred_mean": reference_mean,
            "challenger_calibration_gap": float(challenger_mean - positive_rate),
            "reference_calibration_gap": float(reference_mean - positive_rate),
            "delta_abs_calibration_gap": float(
                abs(challenger_mean - positive_rate) - abs(reference_mean - positive_rate)
            ),
            "mean_abs_pred_shift": float((challenger_pred - reference_pred).abs().mean()),
            "signed_pred_shift": float((challenger_pred - reference_pred).mean()),
            "challenger_hard_case_rate": float(part[challenger_hard_col].mean()),
            "reference_hard_case_rate": float(part[reference_hard_col].mean()),
            "delta_hard_case_rate": float(
                float(part[challenger_hard_col].mean()) - float(part[reference_hard_col].mean())
            ),
        }
        rows.append(row)

    return rows


def _select_records(frame: pd.DataFrame, *, sort_by: str, ascending: bool, top_k: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.sort_values(by=sort_by, ascending=ascending).head(top_k).to_dict(orient="records")


def main() -> int:
    args = parse_args()
    raw_specs = args.oof if args.oof else list(DEFAULT_OOF_SPECS)
    specs = [parse_oof_input_spec(raw) for raw in raw_specs]
    merged_oof, model_columns = load_merged_oof_matrix(specs, target_column=OOF_TARGET_COLUMN)

    reference_weights = _load_weights(args.reference_weights_json)
    challenger_weights = _load_weights(args.challenger_weights_json)

    if any(f"pred_{name}" not in merged_oof.columns for name in challenger_weights):
        missing = [name for name in challenger_weights if f"pred_{name}" not in merged_oof.columns]
        raise ValueError(f"Missing OOF predictions for challenger weights: {missing}")
    if any(f"pred_{name}" not in merged_oof.columns for name in reference_weights):
        missing = [name for name in reference_weights if f"pred_{name}" not in merged_oof.columns]
        raise ValueError(f"Missing OOF predictions for reference weights: {missing}")

    analysis = _build_analysis_frame(args.train_csv, merged_oof)
    analysis[OOF_TARGET_COLUMN] = analysis[OOF_TARGET_COLUMN].astype(int)

    analysis["pred_reference"] = 0.0
    for name, weight in reference_weights.items():
        analysis["pred_reference"] += float(weight) * analysis[f"pred_{name}"].astype("float64")

    analysis["pred_challenger"] = 0.0
    for name, weight in challenger_weights.items():
        analysis["pred_challenger"] += float(weight) * analysis[f"pred_{name}"].astype("float64")

    analysis["loss_reference"] = _logloss_vector(analysis[OOF_TARGET_COLUMN], analysis["pred_reference"])
    analysis["loss_challenger"] = _logloss_vector(analysis[OOF_TARGET_COLUMN], analysis["pred_challenger"])

    hard_quantile = float(args.hard_loss_quantile)
    if not (0.5 < hard_quantile < 1.0):
        raise ValueError("hard-loss-quantile must be in (0.5, 1.0)")

    hard_threshold_reference = float(analysis["loss_reference"].quantile(hard_quantile))
    hard_threshold_challenger = float(analysis["loss_challenger"].quantile(hard_quantile))
    analysis["hard_reference"] = (analysis["loss_reference"] >= hard_threshold_reference).astype("int8")
    analysis["hard_challenger"] = (analysis["loss_challenger"] >= hard_threshold_challenger).astype("int8")

    group_specs = (
        ("contract", "Contract", 1000),
        ("payment_method", "PaymentMethod", 1000),
        ("internet_service", "InternetService", 1000),
        ("paperless", "PaperlessBilling", 1000),
        ("senior", "SeniorCitizen", 1000),
        ("partner", "Partner", 1000),
        ("dependents", "Dependents", 1000),
        ("tenure_bin", "tenure_bin", 1000),
        ("annual_boundary", "annual_boundary_bin", 1000),
        ("contract_boundary", "contract_boundary_bin", 1000),
        ("contract_x_boundary", "contract_x_boundary_bin", 2000),
        ("segment3", "segment3", 5000),
        ("segment5_support", "segment5_support_bucket", 1),
        ("segment5", "segment5", 2000),
    )

    group_rows: list[dict[str, Any]] = []
    for family, column, min_rows in group_specs:
        group_rows.extend(
            _group_metrics(
                analysis,
                family=family,
                column=column,
                min_rows=min_rows,
                challenger_col="pred_challenger",
                reference_col="pred_reference",
                challenger_loss_col="loss_challenger",
                reference_loss_col="loss_reference",
                challenger_hard_col="hard_challenger",
                reference_hard_col="hard_reference",
            )
        )

    groups_df = pd.DataFrame(group_rows)
    groups_out = Path(args.out_groups_csv)
    groups_out.parent.mkdir(parents=True, exist_ok=True)
    groups_df.to_csv(groups_out, index=False)

    hard_cases = groups_df[groups_df["rows"] >= 2000].copy()
    overall_hard_rate = float(analysis["hard_challenger"].mean())
    if not hard_cases.empty and overall_hard_rate > 0.0:
        hard_cases["challenger_hard_case_lift"] = (
            hard_cases["challenger_hard_case_rate"] / overall_hard_rate
        )
    else:
        hard_cases["challenger_hard_case_lift"] = np.nan
    hard_cases = hard_cases.sort_values(
        by=["challenger_hard_case_lift", "rows"],
        ascending=[False, False],
    )
    hard_cases_out = Path(args.out_hard_cases_csv)
    hard_cases_out.parent.mkdir(parents=True, exist_ok=True)
    hard_cases.to_csv(hard_cases_out, index=False)

    overall_reference = _compute_overall_metrics(analysis, "pred_reference")
    overall_challenger = _compute_overall_metrics(analysis, "pred_challenger")

    family_summary_rows: list[dict[str, Any]] = []
    for family, part in groups_df.groupby("family", dropna=False):
        valid_auc = part["delta_auc"].dropna()
        family_summary_rows.append(
            {
                "family": str(family),
                "groups": int(len(part)),
                "rows_sum": int(part["rows"].sum()),
                "weighted_delta_logloss": float(np.average(part["delta_logloss"], weights=part["rows"])),
                "weighted_delta_brier": float(np.average(part["delta_brier"], weights=part["rows"])),
                "weighted_delta_mae": float(np.average(part["delta_mae"], weights=part["rows"])),
                "weighted_delta_auc": (
                    None
                    if valid_auc.empty
                    else float(np.average(valid_auc, weights=part.loc[valid_auc.index, "rows"]))
                ),
                "weighted_mean_abs_pred_shift": float(
                    np.average(part["mean_abs_pred_shift"], weights=part["rows"])
                ),
                "weighted_delta_hard_case_rate": float(
                    np.average(part["delta_hard_case_rate"], weights=part["rows"])
                ),
            }
        )
    family_summary_df = pd.DataFrame(family_summary_rows).sort_values(
        by="weighted_delta_logloss",
        ascending=False,
    )

    top_k = max(int(args.top_k), 1)
    summary = {
        "generated_at_utc": utc_now_iso(),
        "inputs": {
            "train_csv": args.train_csv,
            "oof": raw_specs,
            "reference_weights_json": args.reference_weights_json,
            "challenger_weights_json": args.challenger_weights_json,
            "hard_loss_quantile": hard_quantile,
        },
        "models_loaded": list(model_columns),
        "overall_reference": overall_reference,
        "overall_challenger": overall_challenger,
        "delta_challenger_vs_reference": {
            "auc": float(overall_challenger["auc"] - overall_reference["auc"]),
            "logloss": float(overall_challenger["logloss"] - overall_reference["logloss"]),
            "brier": float(overall_challenger["brier"] - overall_reference["brier"]),
            "mae": float(overall_challenger["mae"] - overall_reference["mae"]),
            "abs_calibration_gap": float(
                abs(overall_challenger["calibration_gap"]) - abs(overall_reference["calibration_gap"])
            ),
        },
        "hard_case_thresholds": {
            "reference_logloss_q": hard_threshold_reference,
            "challenger_logloss_q": hard_threshold_challenger,
            "reference_rate": float(analysis["hard_reference"].mean()),
            "challenger_rate": float(analysis["hard_challenger"].mean()),
        },
        "family_rollup": family_summary_df.to_dict(orient="records"),
        "top_degraded_groups_by_logloss": _select_records(
            groups_df[groups_df["rows"] >= 2000],
            sort_by="logloss_delta_contribution",
            ascending=False,
            top_k=top_k,
        ),
        "top_improved_groups_by_logloss": _select_records(
            groups_df[groups_df["rows"] >= 2000],
            sort_by="logloss_delta_contribution",
            ascending=True,
            top_k=top_k,
        ),
        "worst_groups_in_challenger_by_logloss": _select_records(
            groups_df[groups_df["rows"] >= 2000],
            sort_by="challenger_logloss",
            ascending=False,
            top_k=top_k,
        ),
        "worst_groups_in_challenger_by_abs_calibration_gap": _select_records(
            groups_df.assign(
                challenger_abs_calibration_gap=groups_df["challenger_calibration_gap"].abs()
            )[groups_df["rows"] >= 2000],
            sort_by="challenger_abs_calibration_gap",
            ascending=False,
            top_k=top_k,
        ),
        "hard_case_hotspots": _select_records(
            hard_cases,
            sort_by="challenger_hard_case_lift",
            ascending=False,
            top_k=top_k,
        ),
        "groups_csv_path": str(groups_out),
        "hard_cases_csv_path": str(hard_cases_out),
    }
    write_json(args.out_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
