"""Data-centric audit for label noise, near-duplicates, and hard examples."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import encode_target, load_csv
from .diagnostics import (
    DEFAULT_REFERENCE_OOF_SPECS,
    OOF_TARGET_COLUMN,
    build_family_frame,
    load_merged_oof_matrix,
    load_reference_prediction_frame,
    parse_oof_input_spec,
    utc_now_iso,
)
from .evaluation import binary_auc


DOMINANT_MACROFAMILY = "Electronic check__Month-to-month__Fiber optic"
DEFAULT_V3_OOF_SPEC = "artifacts/reports/validation_protocol_v3_chain_oof.csv#candidate_pred"


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("__missing__")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _normalize_numeric_series(series: pd.Series, *, decimals: int = 6) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    rounded = numeric.round(decimals)
    return rounded.fillna(-999999.0)


def _hash_signature_frame(frame: pd.DataFrame) -> pd.Series:
    hashed = pd.util.hash_pandas_object(frame, index=False).astype("uint64").astype(str)
    return pd.Series(hashed.to_numpy(), index=frame.index, dtype="string")


def _build_exact_signature_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in train_df.columns if col not in {ID_COLUMN, TARGET_COLUMN}]
    normalized: dict[str, pd.Series] = {}
    for column in feature_cols:
        series = train_df[column]
        if pd.api.types.is_numeric_dtype(series):
            normalized[column] = _normalize_numeric_series(series)
        else:
            normalized[column] = _normalize_text_series(series)
    return pd.DataFrame(normalized, index=train_df.index)


def _build_coarse_signature_frame(train_df: pd.DataFrame, family_frame: pd.DataFrame) -> pd.DataFrame:
    coarse = pd.DataFrame(index=train_df.index)
    category_candidates = (
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    )
    for column in category_candidates:
        if column in train_df.columns:
            coarse[column] = _normalize_text_series(train_df[column])

    if "tenure" in train_df.columns:
        tenure = pd.to_numeric(train_df["tenure"], errors="coerce").fillna(-1).astype("int64")
        coarse["tenure_exact"] = tenure.astype(str)
        coarse["tenure_bucket"] = family_frame["tenure_bin"].astype(str).values

    if "MonthlyCharges" in train_df.columns:
        monthly = pd.to_numeric(train_df["MonthlyCharges"], errors="coerce").fillna(-999999.0)
        coarse["monthly_charge_bucket"] = (np.round(monthly / 1.0) * 1.0).round(2)

    if "TotalCharges" in train_df.columns:
        total = pd.to_numeric(train_df["TotalCharges"], errors="coerce").fillna(-999999.0)
        coarse["total_charge_bucket"] = (np.round(total / 10.0) * 10.0).round(2)
    return coarse


def _summarize_duplicate_groups(
    row_frame: pd.DataFrame,
    *,
    signature_kind: str,
    signature_column: str,
) -> pd.DataFrame:
    grouped = row_frame.groupby(signature_column, dropna=False)
    rows: list[dict[str, Any]] = []
    for signature_value, part in grouped:
        if len(part) < 2:
            continue
        labels = sorted(part[OOF_TARGET_COLUMN].astype(int).unique().tolist())
        inconsistent = len(labels) > 1
        sample_ids = part[ID_COLUMN].head(5).astype(int).tolist()
        rows.append(
            {
                "signature_kind": signature_kind,
                "signature": str(signature_value),
                "group_size": int(len(part)),
                "label_unique_count": int(len(labels)),
                "labels": labels,
                "positive_rate": float(part[OOF_TARGET_COLUMN].mean()),
                "inconsistent_labels": bool(inconsistent),
                "segment3_mode": str(part["segment3"].mode(dropna=False).iloc[0]),
                "segment5_mode": str(part["segment5"].mode(dropna=False).iloc[0]),
                "sample_ids": sample_ids,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "signature_kind",
                "signature",
                "group_size",
                "label_unique_count",
                "labels",
                "positive_rate",
                "inconsistent_labels",
                "segment3_mode",
                "segment5_mode",
                "sample_ids",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        by=["inconsistent_labels", "group_size"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _merge_duplicate_flags(
    analysis: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    signature_kind: str,
    signature_column: str,
) -> pd.DataFrame:
    if groups.empty:
        analysis[f"{signature_kind}_group_size"] = 1
        analysis[f"{signature_kind}_inconsistent"] = False
        return analysis

    lookup = groups[[ "signature", "group_size", "inconsistent_labels"]].rename(
        columns={
            "signature": signature_column,
            "group_size": f"{signature_kind}_group_size",
            "inconsistent_labels": f"{signature_kind}_inconsistent",
        }
    )
    out = analysis.merge(lookup, how="left", on=signature_column, validate="many_to_one")
    out[f"{signature_kind}_group_size"] = out[f"{signature_kind}_group_size"].fillna(1).astype(int)
    out[f"{signature_kind}_inconsistent"] = (
        out[f"{signature_kind}_inconsistent"].astype("boolean").fillna(False).astype(bool)
    )
    return out


def _load_teacher_components(
    *,
    oof_specs: Sequence[str] | None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    specs_raw = list(oof_specs or DEFAULT_REFERENCE_OOF_SPECS)
    specs = [parse_oof_input_spec(raw) for raw in specs_raw]
    merged, model_columns = load_merged_oof_matrix(specs, id_column=ID_COLUMN, target_column=OOF_TARGET_COLUMN)
    merged = merged.rename(columns={column: column.replace("pred_", "teacher_") for column in model_columns})
    teacher_cols = [column for column in merged.columns if column.startswith("teacher_")]
    if not teacher_cols:
        raise ValueError("Teacher OOF specs produced zero usable prediction columns.")
    info = {
        "available": True,
        "oof_specs": specs_raw,
        "teacher_columns": teacher_cols,
    }
    return merged[[ID_COLUMN, OOF_TARGET_COLUMN, *teacher_cols]], info


def _build_primary_reason(analysis: pd.DataFrame) -> pd.Series:
    reasons = np.full(len(analysis), "", dtype=object)
    near_mask = analysis["near_duplicate_conflict"].to_numpy(dtype=bool)
    noise_mask = analysis["label_noise_candidate"].to_numpy(dtype=bool)
    hard_mask = analysis["hard_example_stable"].to_numpy(dtype=bool)
    reasons[hard_mask] = "hard_example_stable"
    reasons[noise_mask] = "label_noise_candidate"
    reasons[near_mask] = "near_duplicate_conflict"
    return pd.Series(reasons, index=analysis.index, dtype="string")


def _top_family_table(
    analysis: pd.DataFrame,
    *,
    family_column: str,
    mask_column: str,
    top_n: int,
) -> list[dict[str, Any]]:
    work = analysis.groupby(family_column, dropna=False).agg(
        rows=(ID_COLUMN, "size"),
        suspicious_rows=(mask_column, "sum"),
        mean_logloss=("logloss", "mean"),
        mean_abs_error=("abs_error", "mean"),
    )
    work["suspicious_rate"] = work["suspicious_rows"] / work["rows"].clip(lower=1)
    work = work.sort_values(by=["suspicious_rows", "suspicious_rate"], ascending=[False, False]).head(int(top_n))
    return [
        {
            "family_value": str(idx),
            "rows": int(row["rows"]),
            "suspicious_rows": int(row["suspicious_rows"]),
            "suspicious_rate": float(row["suspicious_rate"]),
            "mean_logloss": float(row["mean_logloss"]),
            "mean_abs_error": float(row["mean_abs_error"]),
        }
        for idx, row in work.iterrows()
    ]


def _fingerprint(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
    return digest.hexdigest()[:12]


def run_label_noise_audit(
    *,
    train_csv_path: str | Path,
    v3_oof_spec: str = DEFAULT_V3_OOF_SPEC,
    teacher_oof_specs: Sequence[str] | None = None,
    high_confidence_upper: float = 0.90,
    high_confidence_lower: float = 0.10,
    hard_loss_quantile: float = 0.995,
    low_disagreement_quantile: float = 0.35,
    high_disagreement_quantile: float = 0.75,
    max_near_duplicate_group_size: int = 5,
    top_n_families: int = 15,
    out_json_path: str | Path | None = None,
    out_rows_csv_path: str | Path | None = None,
    out_duplicate_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a diagnostic-only audit for label noise and near-duplicates."""
    if not (0.5 < high_confidence_upper < 1.0):
        raise ValueError("high_confidence_upper must be in (0.5, 1.0)")
    if not (0.0 < high_confidence_lower < 0.5):
        raise ValueError("high_confidence_lower must be in (0.0, 0.5)")
    if high_confidence_lower >= high_confidence_upper:
        raise ValueError("high_confidence_lower must be less than high_confidence_upper")
    if not (0.0 < hard_loss_quantile < 1.0):
        raise ValueError("hard_loss_quantile must be in (0, 1)")
    if not (0.0 < low_disagreement_quantile < 1.0):
        raise ValueError("low_disagreement_quantile must be in (0, 1)")
    if not (0.0 < high_disagreement_quantile < 1.0):
        raise ValueError("high_disagreement_quantile must be in (0, 1)")
    if int(max_near_duplicate_group_size) < 2:
        raise ValueError("max_near_duplicate_group_size must be >= 2")

    train_df = load_csv(train_csv_path)
    family_frame = build_family_frame(train_df)
    reference_frame, reference_source = load_reference_prediction_frame(
        reference_oof_spec=v3_oof_spec,
        id_column=ID_COLUMN,
        target_column=OOF_TARGET_COLUMN,
    )

    analysis = train_df[[ID_COLUMN, TARGET_COLUMN]].copy()
    analysis = analysis.merge(
        reference_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "reference_pred"]],
        how="inner",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    analysis = analysis.merge(
        family_frame[[ID_COLUMN, "segment3", "segment5", "tenure_bin"]],
        how="inner",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if len(analysis) != len(train_df):
        raise ValueError("v3 OOF coverage does not match train.csv rows.")

    y_true = encode_target(analysis[TARGET_COLUMN]).astype("int8")
    oof_target = analysis[OOF_TARGET_COLUMN].astype("int8")
    if not np.array_equal(y_true.to_numpy(), oof_target.to_numpy()):
        raise ValueError("v3 OOF target does not align with train.csv labels.")

    analysis = analysis.drop(columns=[TARGET_COLUMN, OOF_TARGET_COLUMN]).rename(columns={"reference_pred": "v3_pred"})
    analysis[OOF_TARGET_COLUMN] = y_true.values
    pred = analysis["v3_pred"].astype("float64").clip(1e-6, 1.0 - 1e-6)
    analysis["abs_error"] = np.abs(analysis[OOF_TARGET_COLUMN].astype("float64") - pred)
    analysis["logloss"] = -(
        analysis[OOF_TARGET_COLUMN].astype("float64") * np.log(pred)
        + (1.0 - analysis[OOF_TARGET_COLUMN].astype("float64")) * np.log(1.0 - pred)
    )
    analysis["prediction_margin"] = np.abs(pred - 0.5)
    analysis["high_confidence_wrong"] = (
        ((analysis[OOF_TARGET_COLUMN] == 0) & pred.ge(float(high_confidence_upper)))
        | ((analysis[OOF_TARGET_COLUMN] == 1) & pred.le(float(high_confidence_lower)))
    )

    teacher_info: dict[str, Any] = {"available": False}
    try:
        teacher_frame, teacher_info = _load_teacher_components(oof_specs=teacher_oof_specs)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        teacher_frame = None
        teacher_info = {"available": False, "error": str(exc)}

    if teacher_frame is not None:
        teacher_feature_cols = [column for column in teacher_frame.columns if column.startswith("teacher_")]
        teacher_frame = teacher_frame.rename(columns={OOF_TARGET_COLUMN: "teacher_target"}).copy()
        analysis = analysis.merge(teacher_frame, how="left", on=ID_COLUMN, validate="one_to_one")
        if len(analysis) != len(train_df):
            raise ValueError("Teacher OOF merge changed row coverage unexpectedly.")
        if analysis["teacher_target"].isna().any():
            missing = int(analysis["teacher_target"].isna().sum())
            raise ValueError(f"Teacher OOF coverage is incomplete: {missing} rows missing.")
        if not np.array_equal(
            analysis["teacher_target"].astype("int8").to_numpy(),
            analysis[OOF_TARGET_COLUMN].astype("int8").to_numpy(),
        ):
            raise ValueError("Teacher OOF target does not align with train.csv labels.")
        analysis = analysis.drop(columns=["teacher_target"])
        teacher_matrix = analysis[teacher_feature_cols].to_numpy(dtype="float64")
        analysis["teacher_mean"] = teacher_matrix.mean(axis=1)
        analysis["teacher_std"] = teacher_matrix.std(axis=1)
        analysis["teacher_range"] = teacher_matrix.max(axis=1) - teacher_matrix.min(axis=1)
        analysis["teacher_unanimous_side"] = (
            np.all(teacher_matrix >= 0.5, axis=1) | np.all(teacher_matrix < 0.5, axis=1)
        )
        low_std_threshold = float(analysis["teacher_std"].quantile(float(low_disagreement_quantile)))
        high_std_threshold = float(analysis["teacher_std"].quantile(float(high_disagreement_quantile)))
    else:
        analysis["teacher_mean"] = np.nan
        analysis["teacher_std"] = np.nan
        analysis["teacher_range"] = np.nan
        analysis["teacher_unanimous_side"] = False
        low_std_threshold = float("nan")
        high_std_threshold = float("nan")

    exact_signature_frame = _build_exact_signature_frame(train_df)
    coarse_signature_frame = _build_coarse_signature_frame(train_df, family_frame)
    analysis["exact_signature"] = _hash_signature_frame(exact_signature_frame)
    analysis["coarse_signature"] = _hash_signature_frame(coarse_signature_frame)

    exact_groups = _summarize_duplicate_groups(analysis, signature_kind="exact", signature_column="exact_signature")
    coarse_groups = _summarize_duplicate_groups(analysis, signature_kind="coarse", signature_column="coarse_signature")

    analysis = _merge_duplicate_flags(
        analysis,
        exact_groups,
        signature_kind="exact",
        signature_column="exact_signature",
    )
    analysis = _merge_duplicate_flags(
        analysis,
        coarse_groups,
        signature_kind="coarse",
        signature_column="coarse_signature",
    )
    analysis["near_duplicate_conflict"] = analysis["exact_inconsistent"] | (
        analysis["coarse_inconsistent"] & analysis["coarse_group_size"].le(int(max_near_duplicate_group_size))
    )

    if teacher_frame is not None:
        analysis["label_noise_candidate"] = (
            analysis["high_confidence_wrong"]
            & analysis["teacher_unanimous_side"]
            & analysis["teacher_std"].le(low_std_threshold)
        )
        instability_proxy = analysis["teacher_std"].ge(high_std_threshold) | analysis["prediction_margin"].le(0.15)
    else:
        analysis["label_noise_candidate"] = analysis["high_confidence_wrong"]
        instability_proxy = analysis["prediction_margin"].le(0.15)

    hard_loss_threshold = float(analysis["logloss"].quantile(float(hard_loss_quantile)))
    analysis["hard_example_stable"] = (
        analysis["logloss"].ge(hard_loss_threshold)
        & ~analysis["label_noise_candidate"]
        & ~analysis["near_duplicate_conflict"]
        & instability_proxy
    )

    analysis["suspicious_any"] = (
        analysis["label_noise_candidate"]
        | analysis["near_duplicate_conflict"]
        | analysis["hard_example_stable"]
    )
    analysis["primary_reason"] = _build_primary_reason(analysis)

    suspicious_rows = analysis.loc[analysis["suspicious_any"]].copy()
    suspicious_rows = suspicious_rows.sort_values(
        by=["primary_reason", "logloss", "abs_error"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    duplicate_group_frames = [frame for frame in (exact_groups, coarse_groups) if not frame.empty]
    if duplicate_group_frames:
        duplicate_groups = pd.concat(duplicate_group_frames, axis=0, ignore_index=True)
        duplicate_groups["eligible_near_duplicate_conflict"] = (
            duplicate_groups["signature_kind"].eq("exact")
            | (
                duplicate_groups["signature_kind"].eq("coarse")
                & duplicate_groups["group_size"].le(int(max_near_duplicate_group_size))
            )
        )
        duplicate_groups = duplicate_groups.sort_values(
            by=["eligible_near_duplicate_conflict", "inconsistent_labels", "group_size"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    else:
        duplicate_groups = pd.DataFrame(
            columns=[
                "signature_kind",
                "signature",
                "group_size",
                "label_unique_count",
                "labels",
                "positive_rate",
                "inconsistent_labels",
                "segment3_mode",
                "segment5_mode",
                "sample_ids",
                "eligible_near_duplicate_conflict",
            ]
        )

    dominant_mask = analysis["segment3"].astype(str).eq(DOMINANT_MACROFAMILY)
    suspicious_dom = suspicious_rows["segment3"].astype(str).eq(DOMINANT_MACROFAMILY)
    duplicate_conflict_rows = int(analysis["near_duplicate_conflict"].sum())
    label_noise_rows = int(analysis["label_noise_candidate"].sum())
    hard_rows = int(analysis["hard_example_stable"].sum())
    overall_suspicious_rate = float(analysis["suspicious_any"].mean())
    dominant_suspicious_rate = float(suspicious_dom.sum() / dominant_mask.sum()) if int(dominant_mask.sum()) > 0 else 0.0

    summary: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "train_csv_path": str(train_csv_path),
        "v3_oof_spec": str(v3_oof_spec),
        "teacher_component_source_fingerprint": _fingerprint(list(teacher_info.get("oof_specs", []))) if teacher_info.get("available") else None,
        "reference_source": reference_source,
        "teacher_components": teacher_info,
        "rows": int(len(analysis)),
        "global_v3_auc": float(binary_auc(analysis[OOF_TARGET_COLUMN], analysis["v3_pred"])),
        "thresholds": {
            "high_confidence_upper": float(high_confidence_upper),
            "high_confidence_lower": float(high_confidence_lower),
            "hard_loss_quantile": float(hard_loss_quantile),
            "hard_loss_threshold": hard_loss_threshold,
            "low_disagreement_quantile": float(low_disagreement_quantile),
            "low_teacher_std_threshold": low_std_threshold if np.isfinite(low_std_threshold) else None,
            "high_disagreement_quantile": float(high_disagreement_quantile),
            "high_teacher_std_threshold": high_std_threshold if np.isfinite(high_std_threshold) else None,
            "max_near_duplicate_group_size": int(max_near_duplicate_group_size),
        },
        "suspicion_counts": {
            "suspicious_any": int(analysis["suspicious_any"].sum()),
            "label_noise_candidate": label_noise_rows,
            "near_duplicate_conflict": duplicate_conflict_rows,
            "hard_example_stable": hard_rows,
            "high_confidence_wrong": int(analysis["high_confidence_wrong"].sum()),
        },
        "suspicion_rates": {
            "suspicious_any": overall_suspicious_rate,
            "label_noise_candidate": float(analysis["label_noise_candidate"].mean()),
            "near_duplicate_conflict": float(analysis["near_duplicate_conflict"].mean()),
            "hard_example_stable": float(analysis["hard_example_stable"].mean()),
        },
        "dominant_macrofamily": {
            "family_value": DOMINANT_MACROFAMILY,
            "rows": int(dominant_mask.sum()),
            "suspicious_rows": int(suspicious_dom.sum()),
            "suspicious_rate": dominant_suspicious_rate,
            "label_noise_rows": int(
                analysis.loc[dominant_mask, "label_noise_candidate"].sum()
            ),
            "near_duplicate_conflict_rows": int(
                analysis.loc[dominant_mask, "near_duplicate_conflict"].sum()
            ),
            "hard_example_rows": int(
                analysis.loc[dominant_mask, "hard_example_stable"].sum()
            ),
        },
        "top_segment3_suspicious": _top_family_table(
            analysis,
            family_column="segment3",
            mask_column="suspicious_any",
            top_n=top_n_families,
        ),
        "top_segment5_suspicious": _top_family_table(
            analysis,
            family_column="segment5",
            mask_column="suspicious_any",
            top_n=top_n_families,
        ),
        "top_segment5_label_noise": _top_family_table(
            analysis,
            family_column="segment5",
            mask_column="label_noise_candidate",
            top_n=top_n_families,
        ),
        "top_segment5_duplicate_conflicts": _top_family_table(
            analysis,
            family_column="segment5",
            mask_column="near_duplicate_conflict",
            top_n=top_n_families,
        ),
        "duplicate_groups": {
            "exact_groups_with_size_gt_1": int(len(exact_groups)),
            "exact_groups_with_label_conflict": int(exact_groups["inconsistent_labels"].sum()) if not exact_groups.empty else 0,
            "coarse_groups_with_size_gt_1": int(len(coarse_groups)),
            "coarse_groups_with_label_conflict": int(coarse_groups["inconsistent_labels"].sum()) if not coarse_groups.empty else 0,
            "eligible_conflict_groups": int(duplicate_groups["eligible_near_duplicate_conflict"].sum()) if not duplicate_groups.empty else 0,
        },
        "top_duplicate_conflict_groups": duplicate_groups.loc[
            duplicate_groups["eligible_near_duplicate_conflict"] & duplicate_groups["inconsistent_labels"]
        ].head(int(top_n_families)).to_dict(orient="records") if not duplicate_groups.empty else [],
    }

    if duplicate_conflict_rows >= max(label_noise_rows * 3, 1):
        driver_text = "near-duplicates/cohortes casi repetidas"
        next_action = (
            "Auditar manualmente y con reglas mas finas los grupos `coarse` pequenos e inconsistentes "
            "antes de probar reweight o exclusion."
        )
    else:
        driver_text = "label noise de alta confianza"
        next_action = (
            "Inspeccionar primero los casos de alta confianza erronea con bajo desacuerdo del teacher "
            "antes de probar filtrado o reweight."
        )

    if dominant_suspicious_rate >= overall_suspicious_rate * 1.5:
        concentration_text = (
            "La sospecha se concentra de forma material en la macrofamilia dominante "
            "`Electronic check / Month-to-month / Fiber optic`."
        )
    else:
        concentration_text = (
            "La sospecha no se concentra de forma desproporcionada en la macrofamilia dominante; "
            "conviene revisar tambien familias secundarias del top segment3."
        )

    summary["operational_conclusion"] = {
        "headline": f"La senal dominante del audit apunta mas a {driver_text} que a la otra clase principal.",
        "family_concentration": concentration_text,
        "recommended_next_action": next_action,
        "do_not_do_yet": (
            "No filtrar globalmente filas ni reentrenar con drops masivos hasta separar near-duplicate real de "
            "cohorte legitima repetida."
        ),
    }

    suspicious_output_columns = [
        ID_COLUMN,
        OOF_TARGET_COLUMN,
        "v3_pred",
        "abs_error",
        "logloss",
        "prediction_margin",
        "segment3",
        "segment5",
        "tenure_bin",
        "high_confidence_wrong",
        "label_noise_candidate",
        "near_duplicate_conflict",
        "hard_example_stable",
        "primary_reason",
        "exact_group_size",
        "exact_inconsistent",
        "coarse_group_size",
        "coarse_inconsistent",
        "teacher_mean",
        "teacher_std",
        "teacher_range",
        "teacher_unanimous_side",
    ]
    suspicious_export = suspicious_rows[suspicious_output_columns].copy()

    if out_rows_csv_path is not None:
        out_rows = ensure_parent_dir(out_rows_csv_path)
        suspicious_export.to_csv(out_rows, index=False)
    if out_duplicate_csv_path is not None:
        out_dup = ensure_parent_dir(out_duplicate_csv_path)
        duplicate_groups.to_csv(out_dup, index=False)
    if out_json_path is not None:
        write_json(out_json_path, summary)
    return summary
