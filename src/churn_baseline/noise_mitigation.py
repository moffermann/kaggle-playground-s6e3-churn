"""Minimal mitigation harness for near-duplicate minority rows."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import CatBoostHyperParams, ID_COLUMN
from .data import infer_categorical_columns, load_csv
from .diagnostics import OOF_TARGET_COLUMN, build_family_frame, utc_now_iso
from .evaluation import binary_auc
from .feature_engineering import normalize_feature_blocks
from .modeling import best_iteration_or_default, fit_with_validation, predict_proba
from .noise_audit import DOMINANT_MACROFAMILY
from .pipeline import _build_stratify_labels, _prepare_train_matrix, _transform_pair_with_stateful_blocks


MITIGATION_DOWNWEIGHT = "downweight"
MITIGATION_DROP = "drop"
SUPPORTED_MITIGATION_MODES = (MITIGATION_DOWNWEIGHT, MITIGATION_DROP)


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("__missing__")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _hash_signature_frame(frame: pd.DataFrame) -> pd.Series:
    hashed = pd.util.hash_pandas_object(frame, index=False).astype("uint64").astype(str)
    return pd.Series(hashed.to_numpy(), index=frame.index, dtype="string")


def _build_mitigation_signature_frame(train_df: pd.DataFrame, family_frame: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(index=train_df.index)
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
            work[column] = _normalize_text_series(train_df[column])

    if "tenure" in train_df.columns:
        work["tenure_exact"] = pd.to_numeric(train_df["tenure"], errors="coerce").fillna(-1).astype("int64").astype(str)
    if "MonthlyCharges" in train_df.columns:
        monthly = pd.to_numeric(train_df["MonthlyCharges"], errors="coerce").fillna(-999999.0)
        work["monthly_charge_bucket"] = (np.round(monthly / 1.0) * 1.0).round(2)
    if "TotalCharges" in train_df.columns:
        total = pd.to_numeric(train_df["TotalCharges"], errors="coerce").fillna(-999999.0)
        work["total_charge_bucket"] = (np.round(total / 10.0) * 10.0).round(2)

    work["segment3"] = family_frame["segment3"].astype(str).values
    work["segment5"] = family_frame["segment5"].astype(str).values
    return work


def derive_fold_local_suspects(
    train_fold_df: pd.DataFrame,
    y_train_fold: pd.Series,
    *,
    dominant_only: bool,
    dominant_family_value: str,
    min_group_size: int,
    max_group_size: int,
    majority_share_min: float,
) -> tuple[pd.Series, dict[str, Any]]:
    """Flag minority-label rows inside small mixed coarse duplicate groups.

    Returns:
        suspects:
            Boolean Series indexed like ``train_fold_df`` where ``True`` marks
            minority-label rows selected by the local mitigation rule.
        metadata:
            Summary dict with group counts, flagged row counts, rule params, and
            a ``top_groups`` preview for traceability.
    """
    if int(min_group_size) < 2:
        raise ValueError("min_group_size must be >= 2")
    if int(max_group_size) < int(min_group_size):
        raise ValueError("max_group_size must be >= min_group_size")
    if not (0.5 < float(majority_share_min) <= 1.0):
        raise ValueError("majority_share_min must be in (0.5, 1.0]")

    family_frame = build_family_frame(train_fold_df)
    signature_frame = _build_mitigation_signature_frame(train_fold_df, family_frame)
    signature = _hash_signature_frame(signature_frame)

    work = pd.DataFrame(
        {
            ID_COLUMN: train_fold_df[ID_COLUMN].values,
            OOF_TARGET_COLUMN: y_train_fold.astype("int8").values,
            "segment3": family_frame["segment3"].astype(str).values,
            "segment5": family_frame["segment5"].astype(str).values,
            "signature": signature.values,
        },
        index=train_fold_df.index,
    )
    suspects = pd.Series(False, index=work.index, dtype=bool, name="suspect_row")
    group_rows: list[dict[str, Any]] = []
    examined_groups = 0
    eligible_groups = 0
    flagged_groups = 0
    filtered_out_by_macro = 0
    for signature_value, part in work.groupby("signature", dropna=False):
        examined_groups += 1
        if dominant_only and str(part["segment3"].iloc[0]) != str(dominant_family_value):
            filtered_out_by_macro += 1
            continue
        size = int(len(part))
        if size < int(min_group_size) or size > int(max_group_size):
            continue
        label_counts = part[OOF_TARGET_COLUMN].value_counts(dropna=False).sort_values(ascending=False)
        if len(label_counts) < 2:
            continue
        majority_label = int(label_counts.index[0])
        majority_share = float(label_counts.iloc[0] / size)
        if majority_share < float(majority_share_min):
            continue
        eligible_groups += 1
        minority_mask = part[OOF_TARGET_COLUMN].astype(int).ne(majority_label)
        if not minority_mask.any():
            continue
        suspects.loc[part.index[minority_mask]] = True
        flagged_groups += 1
        group_rows.append(
            {
                "signature": str(signature_value),
                "group_size": size,
                "majority_label": majority_label,
                "majority_share": majority_share,
                "minority_rows": int(minority_mask.sum()),
                "segment3_mode": str(part["segment3"].mode(dropna=False).iloc[0]),
                "segment5_mode": str(part["segment5"].mode(dropna=False).iloc[0]),
            }
        )

    metadata = {
        "examined_groups": int(examined_groups),
        "eligible_groups": int(eligible_groups),
        "flagged_groups": int(flagged_groups),
        "flagged_rows": int(suspects.sum()),
        "dominant_only": bool(dominant_only),
        "dominant_family_value": str(dominant_family_value),
        "filtered_out_by_macro": int(filtered_out_by_macro),
        "min_group_size": int(min_group_size),
        "max_group_size": int(max_group_size),
        "majority_share_min": float(majority_share_min),
        "top_groups": group_rows[:10],
    }
    return suspects, metadata


def run_noise_mitigation_smoke(
    *,
    train_csv_path: str,
    params: CatBoostHyperParams,
    feature_blocks: Sequence[str],
    folds: int,
    seed: int,
    early_stopping_rounds: int,
    verbose: int,
    stratify_mode: str,
    mitigation_mode: str,
    suspect_weight: float,
    dominant_only: bool,
    dominant_family_value: str = DOMINANT_MACROFAMILY,
    min_group_size: int = 3,
    max_group_size: int = 5,
    majority_share_min: float = 0.75,
) -> dict[str, Any]:
    """Run a smoke CatBoost CV with fold-local near-duplicate mitigation.

    Returns a metrics dict with scalar CV summaries, ``fold_metrics`` per fold,
    mitigation rule params, and an ``oof_pred`` Series used by the CLI to build
    direct-vs-v3 analysis artifacts.
    """
    mode = str(mitigation_mode).strip().lower()
    if mode not in SUPPORTED_MITIGATION_MODES:
        raise ValueError(f"Unsupported mitigation_mode '{mitigation_mode}'. Supported: {SUPPORTED_MITIGATION_MODES}")
    if not (0.0 < float(suspect_weight) <= 1.0):
        raise ValueError("suspect_weight must be in (0, 1]")

    train_df = load_csv(train_csv_path)
    normalized_blocks, _, stateful_blocks, x, y = _prepare_train_matrix(train_df, feature_blocks)
    aligned_train_df = train_df.loc[x.index].copy()
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    stratify_labels = _build_stratify_labels(x, y, stratify_mode=stratify_mode, min_count=int(folds))

    oof_pred = pd.Series(index=x.index, dtype="float64", name="oof_pred")
    fold_rows: list[dict[str, Any]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, stratify_labels), start=1):
        raw_train_fold = aligned_train_df.iloc[train_idx].copy()
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        suspect_rows, suspect_meta = derive_fold_local_suspects(
            raw_train_fold,
            y_train,
            dominant_only=dominant_only,
            dominant_family_value=dominant_family_value,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            majority_share_min=majority_share_min,
        )

        if mode == MITIGATION_DROP:
            keep_mask = ~suspect_rows
            x_train_fit = x_train.loc[keep_mask]
            y_train_fit = y_train.loc[keep_mask]
            sample_weight_train = None
        else:
            keep_mask = pd.Series(True, index=x_train.index, dtype=bool)
            x_train_fit = x_train
            y_train_fit = y_train
            sample_weight_train = pd.Series(1.0, index=x_train.index, dtype="float64")
            sample_weight_train.loc[suspect_rows] = float(suspect_weight)

        x_train_fit, x_valid_fit = _transform_pair_with_stateful_blocks(x_train_fit, x_valid, stateful_blocks)
        cat_columns = infer_categorical_columns(x_train_fit)

        fold_params = CatBoostHyperParams(
            iterations=params.iterations,
            learning_rate=params.learning_rate,
            depth=params.depth,
            l2_leaf_reg=params.l2_leaf_reg,
            random_seed=seed,
            loss_function=params.loss_function,
            eval_metric=params.eval_metric,
            monotone_constraints=params.monotone_constraints,
        )

        model = fit_with_validation(
            x_train=x_train_fit,
            y_train=y_train_fit,
            x_valid=x_valid_fit,
            y_valid=y_valid,
            cat_columns=cat_columns,
            params=fold_params,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            sample_weight_train=None if sample_weight_train is None else sample_weight_train.loc[x_train_fit.index],
        )

        fold_pred = predict_proba(model, x_valid_fit)
        oof_pred.iloc[valid_idx] = fold_pred.values
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "auc": float(binary_auc(y_valid, fold_pred)),
                "best_iteration": int(model.get_best_iteration()),
                "final_iterations": int(best_iteration_or_default(model, params.iterations)),
                "train_rows_before": int(len(train_idx)),
                "train_rows_after": int(len(x_train_fit)),
                "suspect_rows": int(suspect_rows.sum()),
                "mode": mode,
                "suspect_meta": suspect_meta,
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError("Noise mitigation OOF predictions contain missing values.")

    fold_aucs = [row["auc"] for row in fold_rows]
    flagged_rows = sum(int(row["suspect_rows"]) for row in fold_rows)
    unique_flagged_rows = None
    if fold_rows:
        per_fold_indices = []
        for train_idx, _ in splitter.split(x, stratify_labels):
            raw_train_fold = aligned_train_df.iloc[train_idx].copy()
            y_train = y.iloc[train_idx]
            suspect_rows, _ = derive_fold_local_suspects(
                raw_train_fold,
                y_train,
                dominant_only=dominant_only,
                dominant_family_value=dominant_family_value,
                min_group_size=min_group_size,
                max_group_size=max_group_size,
                majority_share_min=majority_share_min,
            )
            per_fold_indices.extend(raw_train_fold.loc[suspect_rows, ID_COLUMN].astype(int).tolist())
        unique_flagged_rows = int(len(set(per_fold_indices)))

    return {
        "generated_at_utc": utc_now_iso(),
        "train_csv_path": str(train_csv_path),
        "feature_blocks": list(normalize_feature_blocks(feature_blocks)),
        "mitigation_mode": mode,
        "suspect_weight": float(suspect_weight),
        "dominant_only": bool(dominant_only),
        "dominant_family_value": str(dominant_family_value),
        "min_group_size": int(min_group_size),
        "max_group_size": int(max_group_size),
        "majority_share_min": float(majority_share_min),
        "rows": int(len(train_df)),
        "folds": int(folds),
        "seed": int(seed),
        "cv_mean_auc": float(np.mean(fold_aucs)),
        "cv_std_auc": float(np.std(fold_aucs)),
        "oof_auc": float(binary_auc(y, oof_pred)),
        "oof_pred": oof_pred,
        "flagged_rows_total_across_folds": int(flagged_rows),
        "flagged_unique_row_ids": unique_flagged_rows,
        "fold_metrics": fold_rows,
    }
