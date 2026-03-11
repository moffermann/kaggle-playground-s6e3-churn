"""Diagnostics utilities for CV-vs-leaderboard gap analysis."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from .artifacts import write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import encode_target, infer_categorical_columns, load_csv, prepare_test_features, prepare_train_features
from .evaluation import binary_auc
from .feature_engineering import (
    apply_feature_engineering,
    normalize_feature_blocks,
    partition_feature_blocks,
)
from .pipeline import _transform_pair_with_stateful_blocks


PASS_STATUS = "PASS"
WARN_STATUS = "WARN"
FAIL_STATUS = "FAIL"
OOF_TARGET_COLUMN = "target"


@dataclass(frozen=True)
class OOFInputSpec:
    """Input descriptor for OOF artifact."""

    name: str
    path: str
    prediction_column: str | None = None


DEFAULT_REFERENCE_OOF_SPECS: tuple[str, ...] = (
    "cb=artifacts/reports/train_cv_multiseed_gate_s5_hiiter_oof.csv#oof_ensemble",
    "xgb=artifacts/reports/train_xgboost_cv_multiseed_hiiter_oof.csv#oof_ensemble",
    "lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred",
    "r=artifacts/reports/fe_blockR_hiiter_oof.csv#oof_pred",
    "rv=artifacts/reports/fe_blockRV_hiiter_oof.csv#oof_pred",
)

SUPPORTED_FAMILY_LEVELS = ("segment3", "segment5")


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_feature_blocks_arg(raw: str) -> list[str]:
    """Parse CLI feature block string into normalized feature blocks."""
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def make_check(name: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    """Build a structured check row."""
    if status not in {PASS_STATUS, WARN_STATUS, FAIL_STATUS}:
        raise ValueError(f"Unsupported check status '{status}'")
    return {"name": name, "status": status, "details": details}


def summarize_checks(checks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate check statuses and counters."""
    counters = {PASS_STATUS: 0, WARN_STATUS: 0, FAIL_STATUS: 0}
    for item in checks:
        status = str(item.get("status", "")).upper()
        if status in counters:
            counters[status] += 1

    if counters[FAIL_STATUS] > 0:
        overall = FAIL_STATUS
    elif counters[WARN_STATUS] > 0:
        overall = WARN_STATUS
    else:
        overall = PASS_STATUS

    return {
        "overall_status": overall,
        "check_counts": {
            "pass": int(counters[PASS_STATUS]),
            "warn": int(counters[WARN_STATUS]),
            "fail": int(counters[FAIL_STATUS]),
            "total": int(sum(counters.values())),
        },
    }


def load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    """Load JSON file if path exists; return None otherwise."""
    input_path = Path(path)
    if not input_path.exists():
        return None
    with input_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Expected JSON object in {input_path}, got {type(payload).__name__}")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Calculate SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: str | Path) -> dict[str, Any]:
    """Collect file metadata and hash if file exists."""
    target = Path(path)
    exists = target.exists()
    data: dict[str, Any] = {
        "path": str(target),
        "exists": bool(exists),
    }
    if not exists:
        return data

    stat = target.stat()
    data["size_bytes"] = int(stat.st_size)
    data["modified_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    data["sha256"] = sha256_file(target)
    return data


def collect_git_context() -> dict[str, Any]:
    """Collect lightweight git context for reproducibility."""

    def _run_git(*args: str) -> str | None:
        command = ["git", *args]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        value = result.stdout.strip()
        return value or None

    return {
        "branch": _run_git("branch", "--show-current"),
        "commit": _run_git("rev-parse", "HEAD"),
        "commit_short": _run_git("rev-parse", "--short", "HEAD"),
        "status_short": _run_git("status", "--short"),
    }


def _psi_from_series(
    train_values: pd.Series,
    test_values: pd.Series,
    bins: int,
    epsilon: float = 1e-9,
) -> float | None:
    """Compute population stability index (PSI) using train quantile bins."""
    train_clean = pd.to_numeric(train_values, errors="coerce").dropna().to_numpy(dtype="float64")
    test_clean = pd.to_numeric(test_values, errors="coerce").dropna().to_numpy(dtype="float64")
    if len(train_clean) == 0 or len(test_clean) == 0:
        return None

    quantiles = np.linspace(0.0, 1.0, int(max(bins, 2)) + 1)
    edges = np.unique(np.quantile(train_clean, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_hist, _ = np.histogram(train_clean, bins=edges)
    test_hist, _ = np.histogram(test_clean, bins=edges)

    train_ratio = train_hist.astype("float64") / float(max(np.sum(train_hist), 1))
    test_ratio = test_hist.astype("float64") / float(max(np.sum(test_hist), 1))
    train_ratio = np.clip(train_ratio, epsilon, None)
    test_ratio = np.clip(test_ratio, epsilon, None)
    return float(np.sum((test_ratio - train_ratio) * np.log(test_ratio / train_ratio)))


def build_numeric_drift_table(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    psi_bins: int = 10,
) -> pd.DataFrame:
    """Build per-column numeric drift summary."""
    rows: list[dict[str, Any]] = []
    common_columns = [col for col in train_features.columns if col in test_features.columns]

    for column in common_columns:
        train_series = train_features[column]
        test_series = test_features[column]
        if not pd.api.types.is_numeric_dtype(train_series):
            continue
        if not pd.api.types.is_numeric_dtype(test_series):
            continue

        train_num = pd.to_numeric(train_series, errors="coerce")
        test_num = pd.to_numeric(test_series, errors="coerce")
        train_non_na = train_num.dropna()
        test_non_na = test_num.dropna()
        if len(train_non_na) == 0 or len(test_non_na) == 0:
            continue

        train_mean = float(train_non_na.mean())
        test_mean = float(test_non_na.mean())
        train_std = float(train_non_na.std(ddof=0))
        test_std = float(test_non_na.std(ddof=0))
        std_ratio = float(test_std / train_std) if train_std > 1e-12 else None

        row = {
            "column": str(column),
            "train_count": int(train_non_na.shape[0]),
            "test_count": int(test_non_na.shape[0]),
            "train_missing_rate": float(train_num.isna().mean()),
            "test_missing_rate": float(test_num.isna().mean()),
            "missing_rate_delta": float(test_num.isna().mean() - train_num.isna().mean()),
            "train_mean": train_mean,
            "test_mean": test_mean,
            "mean_delta": float(test_mean - train_mean),
            "train_std": train_std,
            "test_std": test_std,
            "std_ratio_test_over_train": std_ratio,
            "train_p01": float(train_non_na.quantile(0.01)),
            "train_p50": float(train_non_na.quantile(0.50)),
            "train_p99": float(train_non_na.quantile(0.99)),
            "test_p01": float(test_non_na.quantile(0.01)),
            "test_p50": float(test_non_na.quantile(0.50)),
            "test_p99": float(test_non_na.quantile(0.99)),
            "psi": _psi_from_series(train_non_na, test_non_na, bins=psi_bins),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["column", "psi"])
    table = pd.DataFrame(rows)
    return table.sort_values(by="psi", ascending=False, na_position="last").reset_index(drop=True)


def build_categorical_drift_table(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    max_unseen_examples: int = 10,
) -> pd.DataFrame:
    """Build per-column categorical drift summary."""
    rows: list[dict[str, Any]] = []
    common_columns = [col for col in train_features.columns if col in test_features.columns]

    for column in common_columns:
        train_series = train_features[column]
        test_series = test_features[column]
        if not pd.api.types.is_object_dtype(train_series) and not pd.api.types.is_categorical_dtype(
            train_series
        ):
            continue

        train_norm = train_series.fillna("__MISSING__").astype(str)
        test_norm = test_series.fillna("__MISSING__").astype(str)

        train_freq = train_norm.value_counts(normalize=True)
        test_freq = test_norm.value_counts(normalize=True)

        all_values = sorted(set(train_freq.index) | set(test_freq.index))
        train_probs = train_freq.reindex(all_values, fill_value=0.0).to_numpy(dtype="float64")
        test_probs = test_freq.reindex(all_values, fill_value=0.0).to_numpy(dtype="float64")
        tvd = 0.5 * np.sum(np.abs(train_probs - test_probs))

        unseen_test_values = sorted(set(test_freq.index) - set(train_freq.index))
        rows.append(
            {
                "column": str(column),
                "train_unique": int(train_freq.shape[0]),
                "test_unique": int(test_freq.shape[0]),
                "unseen_in_test_count": int(len(unseen_test_values)),
                "unseen_in_test_values": unseen_test_values[:max_unseen_examples],
                "tvd": float(tvd),
                "train_top_1": str(train_freq.index[0]) if not train_freq.empty else None,
                "train_top_1_share": float(train_freq.iloc[0]) if not train_freq.empty else None,
                "test_top_1": str(test_freq.index[0]) if not test_freq.empty else None,
                "test_top_1_share": float(test_freq.iloc[0]) if not test_freq.empty else None,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["column", "tvd"])
    table = pd.DataFrame(rows)
    return table.sort_values(by="tvd", ascending=False).reset_index(drop=True)


def run_adversarial_validation(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    folds: int = 3,
    random_state: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 6,
    early_stopping_rounds: int = 80,
    sample_frac: float = 0.35,
) -> dict[str, Any]:
    """Run adversarial validation using CatBoost (train-vs-test classifier)."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0, 1]")

    train_work = train_features.copy()
    test_work = test_features.copy()
    if sample_frac < 1.0:
        train_work = train_work.sample(frac=sample_frac, random_state=random_state)
        test_work = test_work.sample(frac=sample_frac, random_state=random_state)

    train_work = train_work.reset_index(drop=True)
    test_work = test_work.reset_index(drop=True)

    combined_x = pd.concat([train_work, test_work], axis=0, ignore_index=True)
    combined_y = np.concatenate(
        [
            np.zeros(len(train_work), dtype="int8"),
            np.ones(len(test_work), dtype="int8"),
        ]
    )

    cat_columns = infer_categorical_columns(combined_x)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    fold_rows = []
    fold_iterations = []
    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(combined_x, combined_y), start=1):
        x_train = combined_x.iloc[train_idx]
        y_train = combined_y[train_idx]
        x_valid = combined_x.iloc[valid_idx]
        y_valid = combined_y[valid_idx]

        model = CatBoostClassifier(
            iterations=int(iterations),
            learning_rate=float(learning_rate),
            depth=int(depth),
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=int(random_state),
            allow_writing_files=False,
        )
        model.fit(
            x_train,
            y_train,
            cat_features=cat_columns,
            eval_set=(x_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=int(early_stopping_rounds),
            verbose=False,
        )
        pred_valid = model.predict_proba(x_valid)[:, 1]
        fold_auc = binary_auc(y_valid, pred_valid)
        best_iter = model.get_best_iteration()
        final_iter = int(best_iter) + 1 if best_iter is not None and best_iter >= 0 else int(iterations)
        fold_iterations.append(max(final_iter, 1))
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "auc": float(fold_auc),
                "best_iteration": int(best_iter) if best_iter is not None else None,
                "final_iterations": int(max(final_iter, 1)),
            }
        )

    aucs = [float(item["auc"]) for item in fold_rows]
    final_iterations = max(int(round(float(np.mean(fold_iterations)))), 1)

    full_model = CatBoostClassifier(
        iterations=int(final_iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=int(random_state),
        allow_writing_files=False,
    )
    full_model.fit(combined_x, combined_y, cat_features=cat_columns, verbose=False)
    importances = full_model.get_feature_importance()

    importance_rows = []
    for idx, column in enumerate(combined_x.columns):
        importance_rows.append(
            {
                "feature": str(column),
                "importance": float(importances[idx]),
            }
        )
    importance_rows = sorted(importance_rows, key=lambda item: float(item["importance"]), reverse=True)

    return {
        "train_sample_rows": int(len(train_work)),
        "test_sample_rows": int(len(test_work)),
        "total_rows": int(len(combined_x)),
        "feature_count": int(combined_x.shape[1]),
        "categorical_column_count": int(len(cat_columns)),
        "categorical_columns": cat_columns,
        "cv_folds": int(folds),
        "cv_fold_metrics": fold_rows,
        "cv_mean_auc": float(np.mean(aucs)),
        "cv_std_auc": float(np.std(aucs)),
        "final_iterations": int(final_iterations),
        "top_feature_importance": importance_rows[:30],
        "params": {
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "depth": int(depth),
            "early_stopping_rounds": int(early_stopping_rounds),
            "sample_frac": float(sample_frac),
            "random_state": int(random_state),
        },
    }


def parse_oof_input_spec(raw: str) -> OOFInputSpec:
    """Parse '<name>=<path>[#<prediction_column>]'."""
    text = raw.strip()
    if "=" not in text:
        raise ValueError(
            f"Invalid --oof value '{raw}'. Expected format: <name>=<path>[#<prediction_column>]"
        )
    name, payload = text.split("=", 1)
    name = name.strip()
    payload = payload.strip()
    if not name:
        raise ValueError(f"Invalid --oof value '{raw}': empty model name")
    if not payload:
        raise ValueError(f"Invalid --oof value '{raw}': empty path")

    path_value, has_column, prediction_column = payload.partition("#")
    path_value = path_value.strip()
    prediction_column = prediction_column.strip() if has_column else None
    if not path_value:
        raise ValueError(f"Invalid --oof value '{raw}': empty path")
    if prediction_column == "":
        prediction_column = None
    return OOFInputSpec(name=name, path=path_value, prediction_column=prediction_column)


def _detect_oof_prediction_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Prediction column '{preferred}' not found in OOF file")
        return preferred
    for candidate in ("oof_pred", "oof_ensemble", "prediction", "pred"):
        if candidate in df.columns:
            return candidate
    oof_like = [column for column in df.columns if str(column).startswith("oof_")]
    if len(oof_like) == 1:
        return oof_like[0]
    if len(oof_like) > 1:
        raise ValueError(f"Multiple oof_* columns found, specify one explicitly: {oof_like}")
    raise ValueError("Could not auto-detect OOF prediction column.")


def load_merged_oof_matrix(
    specs: Sequence[OOFInputSpec],
    *,
    id_column: str = "id",
    target_column: str = OOF_TARGET_COLUMN,
) -> tuple[pd.DataFrame, list[str]]:
    """Load and merge OOF files into one matrix: id, target, pred_<name>..."""
    if not specs:
        raise ValueError("At least one OOF input is required.")

    merged: pd.DataFrame | None = None
    model_columns: list[str] = []
    for spec in specs:
        oof_df = pd.read_csv(spec.path)
        if id_column not in oof_df.columns or target_column not in oof_df.columns:
            raise ValueError(
                f"OOF file '{spec.path}' must contain columns '{id_column}' and '{target_column}'."
            )
        pred_column = _detect_oof_prediction_column(oof_df, preferred=spec.prediction_column)
        out_column = f"pred_{spec.name}"
        part = oof_df[[id_column, target_column, pred_column]].copy()
        part = part.rename(columns={pred_column: out_column})
        if merged is None:
            merged = part
        else:
            merged = merged.merge(
                part,
                how="inner",
                on=[id_column, target_column],
                validate="one_to_one",
            )
        model_columns.append(out_column)

    if merged is None or merged.empty:
        raise ValueError("Merged OOF matrix is empty.")
    return merged, model_columns


def load_reference_prediction_frame(
    *,
    reference_oof_spec: str | None = None,
    oof_specs: Sequence[str] | None = None,
    reference_weights_json: str | Path | None = None,
    id_column: str = ID_COLUMN,
    target_column: str = OOF_TARGET_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load reference predictions from either a direct OOF file or a weighted blend."""
    def _normalize_target_column(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        target_series = out[target_column]
        if pd.api.types.is_numeric_dtype(target_series):
            out[target_column] = pd.to_numeric(target_series, errors="raise").astype("int8")
        else:
            out[target_column] = encode_target(target_series)
        return out

    if reference_oof_spec:
        spec = parse_oof_input_spec(f"reference={reference_oof_spec}")
        merged, _ = load_merged_oof_matrix((spec,), id_column=id_column, target_column=target_column)
        merged = _normalize_target_column(merged)
        reference_col = "pred_reference"
        return (
            merged[[id_column, target_column, reference_col]].rename(columns={reference_col: "reference_pred"}),
            {
                "mode": "direct_oof",
                "reference_oof_spec": reference_oof_spec,
            },
        )

    specs_raw = list(oof_specs or DEFAULT_REFERENCE_OOF_SPECS)
    specs = [parse_oof_input_spec(raw) for raw in specs_raw]
    merged, model_columns = load_merged_oof_matrix(specs, id_column=id_column, target_column=target_column)
    merged = _normalize_target_column(merged)
    if reference_weights_json is None:
        raise ValueError("reference_weights_json is required when reference_oof_spec is not provided.")

    payload = json.loads(Path(reference_weights_json).read_text(encoding="utf-8"))
    weights_raw = payload.get("weights")
    if not isinstance(weights_raw, dict) or not weights_raw:
        raise ValueError(f"Could not find weights in {reference_weights_json}")

    weights: dict[str, float] = {}
    for name, value in weights_raw.items():
        column = f"pred_{name}"
        if column not in merged.columns:
            raise ValueError(
                f"Weight '{name}' from {reference_weights_json} does not match merged OOF columns {model_columns}."
            )
        weights[column] = float(value)

    reference_pred = np.zeros(len(merged), dtype="float64")
    for column, weight in weights.items():
        reference_pred += merged[column].to_numpy(dtype="float64") * float(weight)

    out = merged[[id_column, target_column]].copy()
    out["reference_pred"] = reference_pred
    return (
        out,
        {
            "mode": "weighted_oof_blend",
            "reference_weights_json": str(reference_weights_json),
            "oof_specs": specs_raw,
            "weights": {column.replace("pred_", ""): float(weight) for column, weight in weights.items()},
        },
    )


def build_family_frame(
    frame: pd.DataFrame,
    *,
    include_id: bool = True,
) -> pd.DataFrame:
    """Build reusable family keys from raw train/test rows."""
    required = ("PaymentMethod", "Contract", "InternetService", "PaperlessBilling", "tenure")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot build families; missing columns: {missing}")

    work = frame.copy()
    if TARGET_COLUMN in work.columns:
        work = work.drop(columns=[TARGET_COLUMN])
    if not include_id and ID_COLUMN in work.columns:
        work = work.drop(columns=[ID_COLUMN])

    engineered = apply_feature_engineering(work, feature_blocks=["A"])
    out = pd.DataFrame(index=engineered.index)
    if include_id and ID_COLUMN in frame.columns:
        out[ID_COLUMN] = frame[ID_COLUMN].values
    out["segment3"] = (
        engineered["PaymentMethod"].astype(str)
        + "__"
        + engineered["Contract"].astype(str)
        + "__"
        + engineered["InternetService"].astype(str)
    )
    out["segment5"] = (
        engineered["PaymentMethod"].astype(str)
        + "__"
        + engineered["Contract"].astype(str)
        + "__"
        + engineered["InternetService"].astype(str)
        + "__"
        + engineered["PaperlessBilling"].astype(str)
        + "__"
        + engineered["tenure_bin"].astype(str)
    )
    out["tenure_bin"] = engineered["tenure_bin"].astype(str)
    return out


def summarize_reference_family_metrics(
    analysis_frame: pd.DataFrame,
    *,
    family_level: str,
    test_family_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """Aggregate reference metrics by family."""
    if family_level not in SUPPORTED_FAMILY_LEVELS:
        raise ValueError(f"Unsupported family_level '{family_level}'. Supported: {SUPPORTED_FAMILY_LEVELS}")

    family_column = family_level
    y_true = analysis_frame[OOF_TARGET_COLUMN].astype(int)
    reference_pred = analysis_frame["reference_pred"].astype("float64")
    reference_loss = -(y_true * np.log(reference_pred.clip(1e-6, 1.0 - 1e-6)) + (1.0 - y_true) * np.log((1.0 - reference_pred).clip(1e-6, 1.0 - 1e-6)))
    work = analysis_frame[[family_column]].copy()
    work["target"] = y_true.values
    work["reference_pred"] = reference_pred.values
    work["reference_loss"] = reference_loss.values

    total_rows = float(len(work))
    rows: list[dict[str, Any]] = []
    for family_value, part in work.groupby(family_column, dropna=False):
        positives = int(part["target"].sum())
        family_auc = None
        if part["target"].nunique(dropna=False) >= 2:
            family_auc = float(binary_auc(part["target"], part["reference_pred"]))
        test_rows = None
        if test_family_counts is not None:
            test_rows = int(test_family_counts.get(str(family_value), 0))
        mean_loss = float(part["reference_loss"].mean())
        rows.append(
            {
                "family_level": family_level,
                "family_value": str(family_value),
                "train_rows": int(len(part)),
                "test_rows": test_rows,
                "positive_rows": positives,
                "positive_rate": float(part["target"].mean()),
                "reference_auc": family_auc,
                "reference_logloss": mean_loss,
                "reference_logloss_contribution": float(mean_loss * float(len(part)) / total_rows),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    if "test_rows" in summary.columns:
        test_rows_series = summary["test_rows"].fillna(0).astype("float64")
        summary["test_train_ratio"] = (test_rows_series / summary["train_rows"].clip(lower=1).astype("float64")).astype(
            "float64"
        )
    return summary.sort_values(
        by=["reference_logloss_contribution", "train_rows"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _transform_train_valid_for_generalization(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    feature_blocks: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    normalized_blocks = normalize_feature_blocks(feature_blocks)
    stateless_blocks, stateful_blocks = partition_feature_blocks(normalized_blocks)
    x_train, y_train = prepare_train_features(train_df, drop_id=True, feature_blocks=stateless_blocks)
    x_valid, y_valid = prepare_train_features(valid_df, drop_id=True, feature_blocks=stateless_blocks)
    x_train, x_valid = _transform_pair_with_stateful_blocks(x_train, x_valid, stateful_blocks)

    cat_columns = infer_categorical_columns(x_train)
    return x_train, y_train, x_valid, y_valid, cat_columns


def evaluate_family_leaveout_generalization(
    train_df: pd.DataFrame,
    *,
    family_assignments: pd.Series,
    family_values: Sequence[str],
    feature_blocks: Sequence[str] | None,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    random_seed: int,
) -> pd.DataFrame:
    """Train without selected families and score them as unseen cohorts."""
    if len(train_df) != len(family_assignments):
        raise ValueError("train_df and family_assignments must have the same length.")

    rows: list[dict[str, Any]] = []
    for family_value in family_values:
        valid_mask = family_assignments.astype(str).eq(str(family_value))
        train_part = train_df.loc[~valid_mask].copy()
        valid_part = train_df.loc[valid_mask].copy()
        if valid_part.empty:
            continue

        x_train, y_train, x_valid, y_valid, cat_columns = _transform_train_valid_for_generalization(
            train_part,
            valid_part,
            feature_blocks=feature_blocks,
        )

        model = CatBoostClassifier(
            iterations=int(iterations),
            learning_rate=float(learning_rate),
            depth=int(depth),
            l2_leaf_reg=float(l2_leaf_reg),
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=int(random_seed),
            allow_writing_files=False,
        )
        model.fit(x_train, y_train, cat_features=cat_columns, verbose=False)
        pred_valid = model.predict_proba(x_valid)[:, 1]
        pred_valid_series = pd.Series(pred_valid, index=valid_part.index, dtype="float64")

        auc_value = None
        if y_valid.nunique(dropna=False) >= 2:
            auc_value = float(binary_auc(y_valid, pred_valid_series))
        loss_value = float(
            -(
                y_valid.astype("float64") * np.log(pred_valid_series.clip(1e-6, 1.0 - 1e-6))
                + (1.0 - y_valid.astype("float64")) * np.log((1.0 - pred_valid_series).clip(1e-6, 1.0 - 1e-6))
            ).mean()
        )
        rows.append(
            {
                "family_value": str(family_value),
                "lofo_auc": auc_value,
                "lofo_logloss": loss_value,
                "lofo_rows": int(len(valid_part)),
                "lofo_positive_rate": float(y_valid.mean()),
            }
        )

    return pd.DataFrame(rows)


def run_family_generalization_compass(
    *,
    train_csv_path: str | Path,
    test_csv_path: str | Path,
    family_level: str,
    feature_blocks: Sequence[str] | None,
    top_k_families: int,
    min_train_rows: int,
    min_test_rows: int,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    random_seed: int,
    reference_oof_spec: str | None = None,
    oof_specs: Sequence[str] | None = None,
    reference_weights_json: str | Path | None = None,
    out_json_path: str | Path | None = None,
    out_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build ranked family generalization compass using leave-one-family-out retraining."""
    if family_level not in SUPPORTED_FAMILY_LEVELS:
        raise ValueError(f"Unsupported family_level '{family_level}'. Supported: {SUPPORTED_FAMILY_LEVELS}")

    train_df = load_csv(train_csv_path)
    test_df = load_csv(test_csv_path)
    family_train = build_family_frame(train_df)
    family_test = build_family_frame(test_df)
    reference_frame, reference_source = load_reference_prediction_frame(
        reference_oof_spec=reference_oof_spec,
        oof_specs=oof_specs,
        reference_weights_json=reference_weights_json,
        id_column=ID_COLUMN,
        target_column=OOF_TARGET_COLUMN,
    )

    analysis = train_df[[ID_COLUMN, TARGET_COLUMN]].copy().merge(
        family_train[[ID_COLUMN, "segment3", "segment5"]],
        how="inner",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    analysis = analysis.merge(
        reference_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "reference_pred"]],
        how="inner",
        on=ID_COLUMN,
        validate="one_to_one",
    )
    if analysis.empty:
        raise ValueError("Merged family generalization analysis frame is empty.")
    train_target_encoded = encode_target(analysis[TARGET_COLUMN])
    reference_target = analysis[OOF_TARGET_COLUMN].astype("int8")
    if not np.array_equal(train_target_encoded.to_numpy(dtype="int8"), reference_target.to_numpy(dtype="int8")):
        raise ValueError("Reference OOF target does not align with train.csv target labels.")
    analysis = analysis.drop(columns=[OOF_TARGET_COLUMN]).rename(columns={TARGET_COLUMN: OOF_TARGET_COLUMN})
    analysis[OOF_TARGET_COLUMN] = train_target_encoded.astype("int8").values

    test_family_counts = family_test[family_level].astype(str).value_counts(dropna=False)
    reference_summary = summarize_reference_family_metrics(
        analysis,
        family_level=family_level,
        test_family_counts=test_family_counts,
    )
    gated = reference_summary[
        reference_summary["train_rows"].ge(int(min_train_rows))
        & reference_summary["test_rows"].fillna(0).astype(int).ge(int(min_test_rows))
    ].copy()
    selected = gated.head(int(top_k_families)).copy()

    lofo = evaluate_family_leaveout_generalization(
        train_df=train_df,
        family_assignments=family_train[family_level],
        family_values=selected["family_value"].tolist(),
        feature_blocks=feature_blocks,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
    )

    combined = reference_summary.merge(lofo, how="left", on="family_value", validate="one_to_one")
    if not combined.empty:
        combined["generalization_gap_auc"] = combined["reference_auc"] - combined["lofo_auc"]
        combined["generalization_gap_logloss"] = combined["lofo_logloss"] - combined["reference_logloss"]
        combined["generalization_gap_logloss_contribution"] = (
            combined["generalization_gap_logloss"] * combined["train_rows"].astype("float64") / float(len(train_df))
        )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "train_csv_path": str(train_csv_path),
        "test_csv_path": str(test_csv_path),
        "family_level": family_level,
        "feature_blocks": list(normalize_feature_blocks(feature_blocks)),
        "reference_source": reference_source,
        "selection": {
            "top_k_families": int(top_k_families),
            "min_train_rows": int(min_train_rows),
            "min_test_rows": int(min_test_rows),
            "eligible_family_count": int(len(gated)),
            "evaluated_family_count": int(lofo["family_value"].nunique()) if not lofo.empty else 0,
        },
        "model_params": {
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "depth": int(depth),
            "l2_leaf_reg": float(l2_leaf_reg),
            "random_seed": int(random_seed),
        },
        "top_reference_risk_families": reference_summary.head(min(int(top_k_families), len(reference_summary))).to_dict(
            orient="records"
        ),
        "top_generalization_gap_families": combined.dropna(subset=["generalization_gap_logloss"]).sort_values(
            by=["generalization_gap_logloss_contribution", "generalization_gap_auc"],
            ascending=[False, False],
            na_position="last",
        ).head(min(int(top_k_families), len(combined))).to_dict(orient="records"),
    }

    if out_csv_path is not None:
        out_csv = Path(out_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_csv, index=False)
    if out_json_path is not None:
        write_json(out_json_path, summary)
    return summary


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Project weights to simplex."""
    clipped = np.maximum(weights.astype("float64"), 0.0)
    total = float(np.sum(clipped))
    if total <= 0.0:
        return np.full(shape=clipped.shape, fill_value=1.0 / float(len(clipped)), dtype="float64")
    return clipped / total


def coordinate_descent_weights(
    y_true: np.ndarray,
    pred_matrix: np.ndarray,
    *,
    step: float = 0.02,
    max_rounds: int = 25,
) -> dict[str, Any]:
    """Fit simplex-constrained blend weights with coordinate descent."""
    if pred_matrix.ndim != 2 or pred_matrix.shape[1] < 2:
        raise ValueError("pred_matrix must have shape (n_rows, n_models>=2)")
    if step <= 0.0 or step >= 1.0:
        raise ValueError("step must be in (0, 1)")
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    n_models = pred_matrix.shape[1]
    weights = np.full(shape=n_models, fill_value=1.0 / float(n_models), dtype="float64")
    best_auc = binary_auc(y_true, pred_matrix @ weights)
    rounds_used = 0

    for round_idx in range(max_rounds):
        improved = False
        for focus in range(n_models):
            for direction in (-1.0, 1.0):
                candidate = weights.copy()
                old_focus = float(candidate[focus])
                new_focus = old_focus + float(direction) * float(step)
                if new_focus < 0.0 or new_focus > 1.0:
                    continue

                rest_old = 1.0 - old_focus
                rest_new = 1.0 - new_focus
                candidate[focus] = new_focus
                if rest_old <= 1e-12:
                    fill = rest_new / float(n_models - 1)
                    for idx in range(n_models):
                        if idx != focus:
                            candidate[idx] = fill
                else:
                    scale = rest_new / rest_old
                    for idx in range(n_models):
                        if idx != focus:
                            candidate[idx] = candidate[idx] * scale
                candidate = normalize_weights(candidate)
                auc = binary_auc(y_true, pred_matrix @ candidate)
                if float(auc) > float(best_auc) + 1e-10:
                    weights = candidate
                    best_auc = float(auc)
                    improved = True
        rounds_used = round_idx + 1
        if not improved:
            break

    return {
        "weights": weights,
        "auc": float(best_auc),
        "rounds_used": int(rounds_used),
    }


def _rank_average_predictions(pred_matrix: np.ndarray) -> np.ndarray:
    rank_matrix = np.zeros_like(pred_matrix, dtype="float64")
    for idx in range(pred_matrix.shape[1]):
        rank_matrix[:, idx] = pd.Series(pred_matrix[:, idx]).rank(method="average").to_numpy(
            dtype="float64"
        )
    rank_mean = np.mean(rank_matrix, axis=1)
    return rank_mean / float(len(rank_mean))


def evaluate_ensemble_robustness(
    merged_oof: pd.DataFrame,
    model_columns: Sequence[str],
    *,
    target_column: str = OOF_TARGET_COLUMN,
    repeats: int = 3,
    folds: int = 5,
    random_state: int = 42,
    weighted_step: float = 0.02,
    weighted_rounds: int = 25,
) -> dict[str, Any]:
    """Evaluate equal/rank/weighted ensembles under repeated split validation."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if folds < 2:
        raise ValueError("folds must be >= 2")

    y = merged_oof[target_column].astype(int).to_numpy()
    matrix = merged_oof[list(model_columns)].to_numpy(dtype="float64")
    n_models = matrix.shape[1]
    if n_models < 2:
        raise ValueError("At least 2 model columns are required.")

    equal_weights = np.full(shape=n_models, fill_value=1.0 / float(n_models), dtype="float64")
    equal_full_auc = binary_auc(y, matrix @ equal_weights)
    rank_full_auc = binary_auc(y, _rank_average_predictions(matrix))
    weighted_full = coordinate_descent_weights(
        y,
        matrix,
        step=weighted_step,
        max_rounds=weighted_rounds,
    )

    method_fold_aucs: dict[str, list[float]] = {"equal": [], "rank": [], "weighted": []}
    fold_rows: list[dict[str, Any]] = []
    for rep in range(repeats):
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=int(random_state) + int(rep),
        )
        for fold_number, (fit_idx, valid_idx) in enumerate(splitter.split(matrix, y), start=1):
            fit_x = matrix[fit_idx]
            fit_y = y[fit_idx]
            valid_x = matrix[valid_idx]
            valid_y = y[valid_idx]

            equal_auc = binary_auc(valid_y, valid_x @ equal_weights)
            rank_auc = binary_auc(valid_y, _rank_average_predictions(valid_x))
            weighted_fit = coordinate_descent_weights(
                fit_y,
                fit_x,
                step=weighted_step,
                max_rounds=weighted_rounds,
            )
            weighted_pred = valid_x @ weighted_fit["weights"]
            weighted_auc = binary_auc(valid_y, weighted_pred)

            method_fold_aucs["equal"].append(float(equal_auc))
            method_fold_aucs["rank"].append(float(rank_auc))
            method_fold_aucs["weighted"].append(float(weighted_auc))

            fold_rows.append(
                {
                    "repeat": int(rep + 1),
                    "fold": int(fold_number),
                    "valid_rows": int(len(valid_idx)),
                    "equal_auc": float(equal_auc),
                    "rank_auc": float(rank_auc),
                    "weighted_auc": float(weighted_auc),
                    "weighted_fit_weights": {
                        model_columns[idx]: float(weighted_fit["weights"][idx]) for idx in range(n_models)
                    },
                }
            )

    method_summary = {}
    for method, aucs in method_fold_aucs.items():
        method_summary[method] = {
            "cv_mean_auc": float(np.mean(aucs)),
            "cv_std_auc": float(np.std(aucs)),
            "cv_min_auc": float(np.min(aucs)),
            "cv_max_auc": float(np.max(aucs)),
            "full_auc": (
                float(equal_full_auc)
                if method == "equal"
                else float(rank_full_auc) if method == "rank" else float(weighted_full["auc"])
            ),
            "optimism_full_minus_cv_mean": (
                float(equal_full_auc - np.mean(aucs))
                if method == "equal"
                else float(rank_full_auc - np.mean(aucs))
                if method == "rank"
                else float(weighted_full["auc"] - np.mean(aucs))
            ),
        }

    return {
        "rows": int(len(merged_oof)),
        "model_columns": list(model_columns),
        "model_count": int(n_models),
        "repeats": int(repeats),
        "folds": int(folds),
        "random_state": int(random_state),
        "weighted_fit": {
            "full_data_auc": float(weighted_full["auc"]),
            "full_data_weights": {
                model_columns[idx]: float(weighted_full["weights"][idx]) for idx in range(n_models)
            },
            "rounds_used": int(weighted_full["rounds_used"]),
            "step": float(weighted_step),
            "max_rounds": int(weighted_rounds),
        },
        "methods": method_summary,
        "cv_fold_metrics": fold_rows,
    }
