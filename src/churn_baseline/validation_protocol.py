"""Validation reset gate for candidate OOF artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import encode_target, load_csv
from .diagnostics import (
    DEFAULT_REFERENCE_OOF_SPECS,
    FAIL_STATUS,
    OOF_TARGET_COLUMN,
    PASS_STATUS,
    WARN_STATUS,
    build_family_frame,
    collect_git_context,
    describe_file,
    load_merged_oof_matrix,
    load_reference_prediction_frame,
    make_check,
    parse_oof_input_spec,
    summarize_checks,
    summarize_reference_family_metrics,
    utc_now_iso,
)
from .evaluation import binary_auc


SMOKE_STAGE = "smoke"
MIDCAP_STAGE = "midcap"
SUBMISSION_STAGE = "submission"
SUPPORTED_STAGES = (SMOKE_STAGE, MIDCAP_STAGE, SUBMISSION_STAGE)
SUPPORTED_TARGET_LEVELS = ("segment3", "segment5")
DOMINANT_MACROFAMILY = "Electronic check__Month-to-month__Fiber optic"
TENURE_BUCKET_ORDER = ("0_6", "7_12", "13_24", "25_48", "49_plus")
MIN_DELTA_AUC = 1e-05
MAX_DOMINANT_DELTA_AUC = -2e-04
MAX_TARGET_DELTA_AUC = -1e-04
MAX_TOP_DAMAGE_REL_LOGLOSS = 0.03
MAX_CV_STD_MULTIPLIER = 1.15
MIN_PROMOTABLE_TRAIN_ROWS = 2000
MIN_PROMOTABLE_TEST_ROWS = 800
MIN_LARGE_TRAIN_ROWS = 5000
MIN_LARGE_TEST_ROWS = 2000
MIN_MICRO_TRAIN_ROWS = 1000
MIN_MICRO_TEST_ROWS = 400
TOP_DAMAGE_COUNT = 3


def _logloss_vector(y_true: pd.Series, pred: pd.Series, epsilon: float = 1e-6) -> pd.Series:
    clipped = pred.astype("float64").clip(lower=epsilon, upper=1.0 - epsilon)
    y = y_true.astype("float64")
    return -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))


def _safe_auc(y_true: pd.Series, pred: pd.Series) -> float | None:
    if y_true.nunique(dropna=False) < 2:
        return None
    return float(binary_auc(y_true.astype(int), pred.astype("float64")))


def _normalize_analysis_oof_target(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    target_series = out[OOF_TARGET_COLUMN]
    if pd.api.types.is_numeric_dtype(target_series):
        out[OOF_TARGET_COLUMN] = pd.to_numeric(target_series, errors="raise").astype("int8")
    else:
        out[OOF_TARGET_COLUMN] = encode_target(target_series)
    return out


def _load_candidate_prediction_frame(
    *,
    candidate_oof_spec: str,
    id_column: str = ID_COLUMN,
    target_column: str = OOF_TARGET_COLUMN,
) -> pd.DataFrame:
    spec = parse_oof_input_spec(f"candidate={candidate_oof_spec}")
    merged, _ = load_merged_oof_matrix((spec,), id_column=id_column, target_column=target_column)
    merged = _normalize_analysis_oof_target(merged)
    return merged[[id_column, target_column, "pred_candidate"]].rename(columns={"pred_candidate": "candidate_pred"})


def _load_analysis_oof_frame(analysis_oof_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(analysis_oof_path)
    required = {ID_COLUMN, OOF_TARGET_COLUMN, "reference_pred", "candidate_pred"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{analysis_oof_path} must contain columns {sorted(required)}; missing {missing}")
    return _normalize_analysis_oof_target(frame[[ID_COLUMN, OOF_TARGET_COLUMN, "reference_pred", "candidate_pred"]].copy())


def _extract_cv_std(metrics_path: str | Path | None) -> float | None:
    if metrics_path is None:
        return None
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    for key in ("cv_std_auc", "cv_std", "oof_cv_std"):
        value = payload.get(key)
        if value is None:
            continue
        return float(value)
    return None


def _derive_sister_family_value(
    *,
    target_family_level: str,
    target_family_value: str | None,
) -> str | None:
    if target_family_level != "segment5" or not target_family_value:
        return None
    parts = str(target_family_value).split("__")
    if len(parts) != 5:
        return None
    payment, contract, internet, paperless, tenure_bin = parts
    if tenure_bin in TENURE_BUCKET_ORDER:
        position = TENURE_BUCKET_ORDER.index(tenure_bin)
        if position > 0:
            sibling_bucket = TENURE_BUCKET_ORDER[position - 1]
        elif position + 1 < len(TENURE_BUCKET_ORDER):
            sibling_bucket = TENURE_BUCKET_ORDER[position + 1]
        else:
            sibling_bucket = tenure_bin
        return "__".join((payment, contract, internet, paperless, sibling_bucket))
    sibling_paperless = "No" if paperless == "Yes" else "Yes"
    return "__".join((payment, contract, internet, sibling_paperless, tenure_bin))


def _relative_delta(candidate_value: float, reference_value: float, epsilon: float = 1e-12) -> float:
    denominator = max(abs(float(reference_value)), epsilon)
    return float((float(candidate_value) - float(reference_value)) / denominator)


def _summarize_family_candidate_metrics(
    analysis_frame: pd.DataFrame,
    *,
    family_level: str,
    test_family_counts: pd.Series | None,
) -> pd.DataFrame:
    y_true = analysis_frame[OOF_TARGET_COLUMN].astype(int)
    reference_pred = analysis_frame["reference_pred"].astype("float64")
    candidate_pred = analysis_frame["candidate_pred"].astype("float64")
    reference_loss = _logloss_vector(y_true, reference_pred)
    candidate_loss = _logloss_vector(y_true, candidate_pred)

    work = analysis_frame[[family_level]].copy()
    work["target"] = y_true.values
    work["reference_pred"] = reference_pred.values
    work["candidate_pred"] = candidate_pred.values
    work["reference_loss"] = reference_loss.values
    work["candidate_loss"] = candidate_loss.values
    total_rows = float(len(work))

    rows: list[dict[str, Any]] = []
    for family_value, part in work.groupby(family_level, dropna=False):
        reference_auc = _safe_auc(part["target"], part["reference_pred"])
        candidate_auc = _safe_auc(part["target"], part["candidate_pred"])
        reference_logloss = float(part["reference_loss"].mean())
        candidate_logloss = float(part["candidate_loss"].mean())
        reference_contribution = float(reference_logloss * float(len(part)) / total_rows)
        candidate_contribution = float(candidate_logloss * float(len(part)) / total_rows)
        test_rows = None
        if test_family_counts is not None:
            test_rows = int(test_family_counts.get(str(family_value), 0))
        rows.append(
            {
                "family_level": family_level,
                "family_value": str(family_value),
                "train_rows": int(len(part)),
                "test_rows": test_rows,
                "positive_rate": float(part["target"].mean()),
                "reference_auc": reference_auc,
                "candidate_auc": candidate_auc,
                "delta_auc": None if reference_auc is None or candidate_auc is None else float(candidate_auc - reference_auc),
                "reference_logloss": reference_logloss,
                "candidate_logloss": candidate_logloss,
                "reference_logloss_contribution": reference_contribution,
                "candidate_logloss_contribution": candidate_contribution,
                "relative_logloss_contribution_delta": _relative_delta(candidate_contribution, reference_contribution),
            }
        )

    return pd.DataFrame(rows)


def load_protocol_analysis_frame(
    *,
    train_csv_path: str | Path,
    analysis_oof_path: str | Path | None = None,
    candidate_oof_spec: str | None = None,
    reference_oof_spec: str | None = None,
    oof_specs: Sequence[str] | None = None,
    reference_weights_json: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_df = load_csv(train_csv_path)
    family_frame = build_family_frame(train_df)

    if analysis_oof_path is not None:
        prediction_frame = _load_analysis_oof_frame(analysis_oof_path)
        prediction_source = {
            "mode": "analysis_oof",
            "analysis_oof_path": str(analysis_oof_path),
        }
    else:
        if not candidate_oof_spec:
            raise ValueError("candidate_oof_spec is required when analysis_oof_path is not provided.")
        reference_frame, reference_source = load_reference_prediction_frame(
            reference_oof_spec=reference_oof_spec,
            oof_specs=oof_specs or DEFAULT_REFERENCE_OOF_SPECS,
            reference_weights_json=reference_weights_json,
            id_column=ID_COLUMN,
            target_column=OOF_TARGET_COLUMN,
        )
        candidate_frame = _load_candidate_prediction_frame(candidate_oof_spec=candidate_oof_spec)
        prediction_frame = reference_frame.merge(
            candidate_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "candidate_pred"]],
            how="inner",
            on=[ID_COLUMN, OOF_TARGET_COLUMN],
            validate="one_to_one",
        )
        prediction_source = {
            "mode": "reference_plus_candidate",
            "reference_source": reference_source,
            "candidate_oof_spec": candidate_oof_spec,
        }

    analysis = train_df[[ID_COLUMN, TARGET_COLUMN]].copy().merge(
        family_frame[[ID_COLUMN, "segment3", "segment5"]],
        how="inner",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    analysis = analysis.merge(
        prediction_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "reference_pred", "candidate_pred"]],
        how="inner",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if analysis.empty:
        raise ValueError("Merged validation protocol frame is empty.")
    encoded_target = encode_target(analysis[TARGET_COLUMN])
    oof_target = analysis[OOF_TARGET_COLUMN].astype("int8")
    if not np.array_equal(encoded_target.to_numpy(dtype="int8"), oof_target.to_numpy(dtype="int8")):
        raise ValueError("OOF target does not align with train.csv target labels.")
    analysis = analysis.drop(columns=[TARGET_COLUMN]).copy()
    analysis[OOF_TARGET_COLUMN] = oof_target.astype("int8").values
    return analysis, prediction_source


def evaluate_validation_protocol(
    *,
    train_csv_path: str | Path,
    test_csv_path: str | Path,
    stage: str,
    analysis_oof_path: str | Path | None = None,
    candidate_oof_spec: str | None = None,
    reference_oof_spec: str | None = None,
    oof_specs: Sequence[str] | None = None,
    reference_weights_json: str | Path | None = None,
    target_family_level: str = "segment5",
    target_family_value: str | None = None,
    sister_family_value: str | None = None,
    dominant_family_value: str = DOMINANT_MACROFAMILY,
    candidate_metrics_json: str | Path | None = None,
    reference_metrics_json: str | Path | None = None,
    submission_csv_path: str | Path | None = None,
    out_json_path: str | Path | None = None,
) -> dict[str, Any]:
    stage_value = str(stage).strip().lower()
    if stage_value not in SUPPORTED_STAGES:
        raise ValueError(f"Unsupported stage '{stage}'. Supported: {SUPPORTED_STAGES}")
    if target_family_level not in SUPPORTED_TARGET_LEVELS:
        raise ValueError(f"Unsupported target_family_level '{target_family_level}'. Supported: {SUPPORTED_TARGET_LEVELS}")

    analysis, prediction_source = load_protocol_analysis_frame(
        train_csv_path=train_csv_path,
        analysis_oof_path=analysis_oof_path,
        candidate_oof_spec=candidate_oof_spec,
        reference_oof_spec=reference_oof_spec,
        oof_specs=oof_specs,
        reference_weights_json=reference_weights_json,
    )
    test_df = load_csv(test_csv_path)
    test_family = build_family_frame(test_df)
    segment5_test_counts = test_family["segment5"].astype(str).value_counts(dropna=False)
    segment3_test_counts = test_family["segment3"].astype(str).value_counts(dropna=False)

    y_true = analysis[OOF_TARGET_COLUMN].astype(int)
    reference_pred = analysis["reference_pred"].astype("float64")
    candidate_pred = analysis["candidate_pred"].astype("float64")
    reference_auc = float(binary_auc(y_true, reference_pred))
    candidate_auc = float(binary_auc(y_true, candidate_pred))
    delta_auc = float(candidate_auc - reference_auc)

    segment5_summary = _summarize_family_candidate_metrics(
        analysis,
        family_level="segment5",
        test_family_counts=segment5_test_counts,
    )
    segment3_summary = _summarize_family_candidate_metrics(
        analysis,
        family_level="segment3",
        test_family_counts=segment3_test_counts,
    )

    target_summary = segment5_summary if target_family_level == "segment5" else segment3_summary
    target_family_row = None
    if target_family_value:
        target_match = target_summary.loc[target_summary["family_value"].astype(str).eq(str(target_family_value))]
        if not target_match.empty:
            target_family_row = target_match.iloc[0].to_dict()
    resolved_sister_family = sister_family_value or _derive_sister_family_value(
        target_family_level=target_family_level,
        target_family_value=target_family_value,
    )
    sister_family_row = None
    if resolved_sister_family and target_family_level == "segment5":
        sister_match = segment5_summary.loc[segment5_summary["family_value"].astype(str).eq(str(resolved_sister_family))]
        if not sister_match.empty:
            sister_family_row = sister_match.iloc[0].to_dict()

    dominant_match = segment3_summary.loc[segment3_summary["family_value"].astype(str).eq(str(dominant_family_value))]
    dominant_row = dominant_match.iloc[0].to_dict() if not dominant_match.empty else None

    top_damage = segment5_summary.loc[
        (segment5_summary["train_rows"].astype(int) >= MIN_LARGE_TRAIN_ROWS)
        | (segment5_summary["test_rows"].fillna(0).astype(int) >= MIN_LARGE_TEST_ROWS)
    ].copy()
    if top_damage.empty:
        top_damage = segment5_summary.copy()
    top_damage = top_damage.sort_values(
        by=["reference_logloss_contribution", "train_rows"],
        ascending=[False, False],
    ).head(TOP_DAMAGE_COUNT)

    candidate_cv_std = _extract_cv_std(candidate_metrics_json)
    reference_cv_std = _extract_cv_std(reference_metrics_json)

    checks: list[dict[str, Any]] = []
    checks.append(
        make_check(
            "split_a_delta_auc",
            PASS_STATUS if delta_auc >= MIN_DELTA_AUC else FAIL_STATUS,
            {
                "stage": stage_value,
                "candidate_oof_auc": candidate_auc,
                "reference_oof_auc": reference_auc,
                "delta_oof_auc": delta_auc,
                "min_required": MIN_DELTA_AUC,
            },
        )
    )

    if dominant_row is None:
        checks.append(make_check("split_b_dominant_family", FAIL_STATUS, {"family_value": dominant_family_value, "reason": "missing"}))
    else:
        dominant_delta_auc = dominant_row.get("delta_auc")
        dominant_status = PASS_STATUS
        if dominant_delta_auc is None or float(dominant_delta_auc) < MAX_DOMINANT_DELTA_AUC:
            dominant_status = FAIL_STATUS
        checks.append(
            make_check(
                "split_b_dominant_family",
                dominant_status,
                {
                    "family_value": dominant_family_value,
                    "delta_auc": dominant_delta_auc,
                    "min_allowed": MAX_DOMINANT_DELTA_AUC,
                },
            )
        )

    top_damage_breaks: list[dict[str, Any]] = []
    for row in top_damage.to_dict(orient="records"):
        relative_delta = float(row["relative_logloss_contribution_delta"])
        if relative_delta > MAX_TOP_DAMAGE_REL_LOGLOSS:
            top_damage_breaks.append(
                {
                    "family_value": row["family_value"],
                    "relative_logloss_contribution_delta": relative_delta,
                }
            )
    checks.append(
        make_check(
            "split_b_top_damage_families",
            PASS_STATUS if not top_damage_breaks else FAIL_STATUS,
            {
                "max_allowed_relative_delta": MAX_TOP_DAMAGE_REL_LOGLOSS,
                "top_damage_families": top_damage[["family_value", "reference_logloss_contribution"]].to_dict(orient="records"),
                "breaks": top_damage_breaks,
            },
        )
    )

    positive_improvements = segment5_summary.loc[
        segment5_summary["relative_logloss_contribution_delta"].astype("float64") < 0.0
    ].copy()
    micro_only = False
    if not positive_improvements.empty:
        micro_mask = (
            positive_improvements["train_rows"].astype(int) < MIN_MICRO_TRAIN_ROWS
        ) | (
            positive_improvements["test_rows"].fillna(0).astype(int) < MIN_MICRO_TEST_ROWS
        )
        micro_only = bool(micro_mask.all())
    checks.append(
        make_check(
            "micro_family_dependence",
            FAIL_STATUS if micro_only else PASS_STATUS,
            {
                "micro_only_improvement": micro_only,
                "improved_family_count": int(len(positive_improvements)),
                "micro_thresholds": {
                    "train_rows_lt": MIN_MICRO_TRAIN_ROWS,
                    "test_rows_lt": MIN_MICRO_TEST_ROWS,
                },
            },
        )
    )

    if stage_value in {MIDCAP_STAGE, SUBMISSION_STAGE}:
        if candidate_cv_std is None or reference_cv_std is None:
            checks.append(
                make_check(
                    "midcap_cv_std",
                    FAIL_STATUS,
                    {
                        "reason": "candidate_metrics_json and reference_metrics_json with cv std are required",
                        "candidate_cv_std": candidate_cv_std,
                        "reference_cv_std": reference_cv_std,
                    },
                )
            )
        else:
            checks.append(
                make_check(
                    "midcap_cv_std",
                    PASS_STATUS if candidate_cv_std <= (reference_cv_std * MAX_CV_STD_MULTIPLIER) else FAIL_STATUS,
                    {
                        "candidate_cv_std": candidate_cv_std,
                        "reference_cv_std": reference_cv_std,
                        "max_multiplier": MAX_CV_STD_MULTIPLIER,
                    },
                )
            )

    if target_family_value:
        if target_family_row is None:
            checks.append(
                make_check(
                    "target_family_presence",
                    FAIL_STATUS,
                    {
                        "target_family_level": target_family_level,
                        "target_family_value": target_family_value,
                    },
                )
            )
        else:
            checks.append(
                make_check(
                    "target_family_presence",
                    PASS_STATUS,
                    {
                        "target_family_level": target_family_level,
                        "target_family_value": target_family_value,
                        "train_rows": int(target_family_row["train_rows"]),
                        "test_rows": int(target_family_row["test_rows"] or 0),
                    },
                )
            )
            if stage_value in {MIDCAP_STAGE, SUBMISSION_STAGE}:
                target_delta_auc = target_family_row.get("delta_auc")
                checks.append(
                    make_check(
                        "target_family_on_mask_auc",
                        PASS_STATUS if target_delta_auc is not None and float(target_delta_auc) >= MAX_TARGET_DELTA_AUC else FAIL_STATUS,
                        {
                            "target_family_level": target_family_level,
                            "target_family_value": target_family_value,
                            "delta_auc": target_delta_auc,
                            "min_allowed": MAX_TARGET_DELTA_AUC,
                        },
                    )
                )
            if stage_value == SUBMISSION_STAGE:
                promotable = (
                    int(target_family_row["train_rows"]) >= MIN_PROMOTABLE_TRAIN_ROWS
                    and int(target_family_row["test_rows"] or 0) >= MIN_PROMOTABLE_TEST_ROWS
                )
                checks.append(
                    make_check(
                        "target_family_promotable_support",
                        PASS_STATUS if promotable else FAIL_STATUS,
                        {
                            "train_rows": int(target_family_row["train_rows"]),
                            "test_rows": int(target_family_row["test_rows"] or 0),
                            "min_train_rows": MIN_PROMOTABLE_TRAIN_ROWS,
                            "min_test_rows": MIN_PROMOTABLE_TEST_ROWS,
                        },
                    )
                )

    if resolved_sister_family and stage_value == SUBMISSION_STAGE and target_family_level == "segment5":
        if sister_family_row is None:
            checks.append(
                make_check(
                    "sister_family_guardrail",
                    FAIL_STATUS,
                    {
                        "sister_family_value": resolved_sister_family,
                        "reason": "missing",
                    },
                )
            )
        else:
            sister_delta_auc = sister_family_row.get("delta_auc")
            checks.append(
                make_check(
                    "sister_family_guardrail",
                    PASS_STATUS if sister_delta_auc is not None and float(sister_delta_auc) >= MAX_TARGET_DELTA_AUC else FAIL_STATUS,
                    {
                        "sister_family_value": resolved_sister_family,
                        "delta_auc": sister_delta_auc,
                        "min_allowed": MAX_TARGET_DELTA_AUC,
                    },
                )
            )

    if stage_value == SUBMISSION_STAGE:
        checks.append(
            make_check(
                "submission_trace_artifacts",
                PASS_STATUS if submission_csv_path else FAIL_STATUS,
                {
                    "submission_csv": None if submission_csv_path is None else describe_file(submission_csv_path),
                    "candidate_metrics_json": None if candidate_metrics_json is None else describe_file(candidate_metrics_json),
                    "reference_metrics_json": None if reference_metrics_json is None else describe_file(reference_metrics_json),
                    "analysis_oof_path": None if analysis_oof_path is None else describe_file(analysis_oof_path),
                },
            )
        )

    verdict = summarize_checks(checks)
    result: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "stage": stage_value,
        "train_csv_path": str(train_csv_path),
        "test_csv_path": str(test_csv_path),
        "prediction_source": prediction_source,
        "target_family_level": target_family_level,
        "target_family_value": target_family_value,
        "resolved_sister_family_value": resolved_sister_family,
        "dominant_family_value": dominant_family_value,
        "overall_metrics": {
            "reference_oof_auc": reference_auc,
            "candidate_oof_auc": candidate_auc,
            "delta_oof_auc": delta_auc,
            "candidate_cv_std": candidate_cv_std,
            "reference_cv_std": reference_cv_std,
        },
        "top_damage_families": top_damage.to_dict(orient="records"),
        "dominant_family": dominant_row,
        "target_family": target_family_row,
        "sister_family": sister_family_row,
        "checks": checks,
        "verdict": verdict,
        "git_context": collect_git_context(),
    }
    if out_json_path is not None:
        write_json(out_json_path, result)
    return result
