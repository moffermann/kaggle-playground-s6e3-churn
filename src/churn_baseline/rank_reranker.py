"""Ranking-aware reranker experiments over the incumbent teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import StratifiedKFold

from .artifacts import ensure_parent_dir, write_json
from .config import CatBoostHyperParams, ID_COLUMN, TARGET_COLUMN
from .data import infer_categorical_columns, load_csv
from .evaluation import binary_auc
from .feature_engineering import normalize_feature_blocks
from .pipeline import (
    STRATIFY_TARGET,
    _build_stratify_labels,
    _prepare_train_matrix,
    _transform_pair_with_stateful_blocks,
    _transform_single_with_stateful_blocks,
    normalize_stratify_mode,
)
from .specialist import _append_reference_features, _normalize_reference_component_frame, build_specialist_mask


PAIR_LOGIT_PAIRWISE = "PairLogitPairwise"
YETI_RANK_PAIRWISE = "YetiRankPairwise"
RANK_RERANKER_LOSSES: tuple[str, ...] = (PAIR_LOGIT_PAIRWISE, YETI_RANK_PAIRWISE)
RANK_RERANKER_QUERY_LEVELS: tuple[str, ...] = (
    "segment5",
    "segment3",
    "contract_internet_tenure",
    "contract_tenure",
    "contract",
    "global",
)
_MIN_LOCAL_RERANK_TRAIN_ROWS = 2000
_MIN_LOCAL_RERANK_VALID_ROWS = 500


def list_rank_reranker_losses() -> tuple[str, ...]:
    """Return supported CatBoost ranker loss functions."""
    return RANK_RERANKER_LOSSES


def list_rank_reranker_query_levels() -> tuple[str, ...]:
    """Return supported hierarchical query levels."""
    return RANK_RERANKER_QUERY_LEVELS


def _build_tenure_bin(tenure_values: pd.Series) -> pd.Series:
    bins = [-np.inf, 6, 12, 24, 48, np.inf]
    labels = ["0_6", "7_12", "13_24", "25_48", "49_plus"]
    return pd.cut(tenure_values, bins=bins, labels=labels).astype(str)


def _align_reference_prediction(
    *,
    train_ids: pd.Series,
    reference_pred: pd.Series,
    index: pd.Index,
) -> pd.Series:
    reference_by_id = pd.Series(reference_pred, copy=True)
    if reference_by_id.index.has_duplicates:
        duplicate_ids = pd.Index(reference_by_id.index[reference_by_id.index.duplicated()]).unique().tolist()[:5]
        raise ValueError(
            "reference_pred must be indexed by unique train ids and represent OOF predictions. "
            f"Duplicate ids found: {duplicate_ids}"
        )
    aligned_reference = train_ids.map(reference_by_id)
    if aligned_reference.isna().any():
        missing_ids = train_ids.loc[aligned_reference.isna()].head(5).tolist()
        raise ValueError(
            "reference_pred must cover every train id with OOF-aligned predictions. "
            f"Missing ids: {missing_ids}"
        )
    return pd.Series(aligned_reference.values, index=index, dtype="float64", name="reference_pred")


def _build_query_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    contract = frame["Contract"].astype(str)
    payment = frame["PaymentMethod"].astype(str)
    internet = frame["InternetService"].astype(str)
    paperless = frame["PaperlessBilling"].astype(str)
    tenure = pd.to_numeric(frame["tenure"], errors="coerce").fillna(0.0)
    tenure_bin = _build_tenure_bin(tenure)

    payment_contract = payment + "__" + contract
    segment3 = payment_contract + "__" + internet
    return pd.DataFrame(
        {
            "segment5": segment3 + "__" + paperless + "__" + tenure_bin,
            "segment3": segment3,
            "contract_internet_tenure": contract + "__" + internet + "__" + tenure_bin,
            "contract_tenure": contract + "__" + tenure_bin,
            "contract": contract,
            "global": pd.Series("__GLOBAL__", index=frame.index, dtype="object"),
        },
        index=frame.index,
    )


def _compute_query_eligibility(
    labels: pd.Series,
    y: pd.Series,
    *,
    min_rows: int,
    min_positive_rows: int,
    min_negative_rows: int,
) -> pd.Series:
    stats = pd.DataFrame({"label": labels.astype(str), "target": y.astype(int)}, index=y.index)
    grouped = stats.groupby("label", dropna=False)["target"].agg(["size", "sum"])
    grouped["neg"] = grouped["size"] - grouped["sum"]
    eligible_labels = grouped.index[
        grouped["size"].ge(int(min_rows))
        & grouped["sum"].ge(int(min_positive_rows))
        & grouped["neg"].ge(int(min_negative_rows))
    ]
    return labels.astype(str).isin(set(eligible_labels))


def _assign_query_groups(
    query_candidates: pd.DataFrame,
    y: pd.Series,
    *,
    min_rows: int,
    min_positive_rows: int,
    min_negative_rows: int,
) -> tuple[np.ndarray, pd.Series, dict[str, int]]:
    selected_labels = pd.Series("__GLOBAL__", index=query_candidates.index, dtype="object")
    selected_levels = pd.Series("global", index=query_candidates.index, dtype="object")
    assigned = pd.Series(False, index=query_candidates.index, dtype=bool)

    hierarchy = ("segment5", "segment3", "contract_internet_tenure", "contract_tenure", "contract")
    for level_name in hierarchy:
        labels = query_candidates[level_name].astype(str)
        eligible = _compute_query_eligibility(
            labels,
            y,
            min_rows=min_rows,
            min_positive_rows=min_positive_rows,
            min_negative_rows=min_negative_rows,
        )
        take_mask = eligible & ~assigned
        selected_labels.loc[take_mask] = level_name + "::" + labels.loc[take_mask]
        selected_levels.loc[take_mask] = level_name
        assigned.loc[take_mask] = True

    group_ids, _ = pd.factorize(selected_labels.astype(str), sort=False)
    level_counts = {str(name): int(count) for name, count in selected_levels.value_counts(dropna=False).items()}
    return group_ids.astype("int64"), selected_levels, level_counts


def _sort_grouped_frame(
    x: pd.DataFrame,
    y: pd.Series,
    group_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    order = np.argsort(group_ids, kind="stable")
    return x.iloc[order], y.iloc[order], np.asarray(group_ids, dtype="int64")[order]


def _build_ranker(
    *,
    params: CatBoostHyperParams,
    loss_function: str,
) -> CatBoostRanker:
    loss_value = str(loss_function).strip()
    if loss_value not in RANK_RERANKER_LOSSES:
        raise ValueError(f"Unsupported loss_function '{loss_function}'. Available: {RANK_RERANKER_LOSSES}")
    return CatBoostRanker(
        iterations=int(params.iterations),
        learning_rate=float(params.learning_rate),
        depth=int(params.depth),
        l2_leaf_reg=float(params.l2_leaf_reg),
        loss_function=loss_value,
        eval_metric="PairAccuracy",
        random_seed=int(params.random_seed),
        allow_writing_files=False,
        verbose=False,
    )


def _best_iteration_or_default(model: CatBoostRanker, default_iterations: int) -> int:
    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter < 0:
        return int(default_iterations)
    return int(best_iter) + 1


def _reorder_reference_by_rank(
    reference_pred: pd.Series,
    ranker_score: pd.Series,
) -> pd.Series:
    reference_values = reference_pred.astype("float64").to_numpy()
    sorted_reference = np.sort(reference_values)
    order = np.argsort(ranker_score.astype("float64").to_numpy(), kind="mergesort")
    reordered = np.empty_like(sorted_reference)
    reordered[order] = sorted_reference
    return pd.Series(reordered, index=reference_pred.index, dtype="float64", name="candidate_score")


def _save_model(model: CatBoostRanker, path: str | Path) -> Path:
    out_path = ensure_parent_dir(path)
    model.save_model(str(out_path))
    return out_path


def _build_sampled_pairs(
    y_sorted: pd.Series,
    group_sorted: np.ndarray,
    *,
    max_pairs_per_group: int,
    random_seed: int,
) -> np.ndarray:
    labels = y_sorted.astype(int).to_numpy()
    groups = np.asarray(group_sorted, dtype="int64")
    if labels.shape[0] != groups.shape[0]:
        raise ValueError("labels and group ids must have the same length")

    rng = np.random.default_rng(int(random_seed))
    pair_chunks: list[np.ndarray] = []
    start = 0
    total_rows = len(labels)
    while start < total_rows:
        end = start + 1
        current_group = groups[start]
        while end < total_rows and groups[end] == current_group:
            end += 1

        local_labels = labels[start:end]
        pos_idx = np.flatnonzero(local_labels == 1) + start
        neg_idx = np.flatnonzero(local_labels == 0) + start
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            pair_count = min(int(max_pairs_per_group), int(len(pos_idx) * len(neg_idx)))
            left = rng.choice(pos_idx, size=pair_count, replace=True)
            right = rng.choice(neg_idx, size=pair_count, replace=True)
            pair_chunks.append(np.column_stack([left, right]).astype("int32"))
        start = end

    if not pair_chunks:
        raise ValueError("No valid ranking pairs could be generated from query groups")
    return np.vstack(pair_chunks)


def run_rank_reranker_cv(
    *,
    train_csv_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
    params: CatBoostHyperParams,
    feature_blocks: Sequence[str] | None,
    reference_pred: pd.Series,
    reference_component_frame: pd.DataFrame | None,
    folds: int,
    random_state: int,
    early_stopping_rounds: int,
    verbose: int,
    loss_function: str,
    alpha_grid: Sequence[float],
    stratify_mode: str = STRATIFY_TARGET,
    min_query_rows: int = 50,
    min_query_positive_rows: int = 3,
    min_query_negative_rows: int = 10,
    max_pairs_per_group: int = 1000,
    include_logit_reference: bool = True,
    preset: str | None = None,
) -> dict[str, Any]:
    """Train a CatBoost ranker over teacher-aware features and scan rank blends."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one value")

    train_df = load_csv(train_csv_path)
    normalized_blocks, _, stateful_blocks, x_base, y = _prepare_train_matrix(train_df, feature_blocks)
    normalized_stratify_mode = normalize_stratify_mode(stratify_mode)
    query_candidates = _build_query_candidate_frame(train_df)
    local_mask = None
    if preset is not None:
        local_mask = build_specialist_mask(train_df, preset).astype(bool)
        local_rows = int(local_mask.sum())
        if local_rows < _MIN_LOCAL_RERANK_TRAIN_ROWS:
            raise ValueError(
                f"Preset '{preset}' selects only {local_rows} rows. Minimum required: {_MIN_LOCAL_RERANK_TRAIN_ROWS}"
            )
        if y.loc[local_mask].nunique() < 2:
            raise ValueError(f"Preset '{preset}' must contain both classes.")
    aligned_reference = _align_reference_prediction(
        train_ids=train_df[ID_COLUMN],
        reference_pred=reference_pred,
        index=x_base.index,
    )
    component_frame = _normalize_reference_component_frame(train_df[ID_COLUMN], reference_component_frame)
    splitter = StratifiedKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(random_state),
    )
    stratify_labels = _build_stratify_labels(
        x_base,
        y,
        stratify_mode=normalized_stratify_mode,
        min_count=max(int(folds), 2),
    )

    ranker_oof = pd.Series(index=x_base.index, dtype="float64", name="ranker_oof_score")
    fold_rows: list[dict[str, Any]] = []
    fold_iterations: list[int] = []
    overall_query_level_counts: dict[str, int] = {}
    disagreement_columns: list[str] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x_base, stratify_labels), start=1):
        x_train = x_base.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x_base.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]
        x_train, x_valid = _transform_pair_with_stateful_blocks(x_train, x_valid, stateful_blocks)

        train_reference = aligned_reference.iloc[train_idx]
        valid_reference = aligned_reference.iloc[valid_idx]
        train_components = component_frame.iloc[train_idx] if component_frame is not None else None
        valid_components = component_frame.iloc[valid_idx] if component_frame is not None else None
        x_train, train_disagreement_columns = _append_reference_features(
            x_train,
            reference_pred=train_reference,
            reference_component_frame=train_components,
            include_logit=include_logit_reference,
        )
        x_valid, valid_disagreement_columns = _append_reference_features(
            x_valid,
            reference_pred=valid_reference,
            reference_component_frame=valid_components,
            include_logit=include_logit_reference,
        )
        disagreement_columns = train_disagreement_columns or valid_disagreement_columns

        train_local_mask = None if local_mask is None else local_mask.iloc[train_idx].to_numpy(dtype=bool)
        valid_local_mask = None if local_mask is None else local_mask.iloc[valid_idx].to_numpy(dtype=bool)
        if train_local_mask is not None:
            local_train_rows = int(np.sum(train_local_mask))
            local_valid_rows = int(np.sum(valid_local_mask))
            if local_train_rows < _MIN_LOCAL_RERANK_TRAIN_ROWS:
                raise ValueError(
                    f"Fold {fold_number} preset '{preset}' train rows {local_train_rows} below minimum."
                )
            if local_valid_rows < _MIN_LOCAL_RERANK_VALID_ROWS:
                raise ValueError(
                    f"Fold {fold_number} preset '{preset}' valid rows {local_valid_rows} below minimum."
                )
            x_train_fit = x_train.iloc[train_local_mask]
            y_train_fit = y_train.iloc[train_local_mask]
            x_valid_fit = x_valid.iloc[valid_local_mask]
            y_valid_fit = y_valid.iloc[valid_local_mask]
            train_reference_eval = train_reference.iloc[train_local_mask]
            valid_reference_eval = valid_reference.iloc[valid_local_mask]
            train_query_frame = query_candidates.iloc[train_idx].iloc[train_local_mask]
            valid_query_frame = query_candidates.iloc[valid_idx].iloc[valid_local_mask]
        else:
            x_train_fit = x_train
            y_train_fit = y_train
            x_valid_fit = x_valid
            y_valid_fit = y_valid
            train_reference_eval = train_reference
            valid_reference_eval = valid_reference
            train_query_frame = query_candidates.iloc[train_idx]
            valid_query_frame = query_candidates.iloc[valid_idx]

        train_query_ids, train_levels, train_level_counts = _assign_query_groups(
            train_query_frame,
            y_train_fit,
            min_rows=min_query_rows,
            min_positive_rows=min_query_positive_rows,
            min_negative_rows=min_query_negative_rows,
        )
        valid_query_ids, valid_levels, valid_level_counts = _assign_query_groups(
            valid_query_frame,
            y_valid_fit,
            min_rows=min_query_rows,
            min_positive_rows=min_query_positive_rows,
            min_negative_rows=min_query_negative_rows,
        )
        for name, count in train_level_counts.items():
            overall_query_level_counts[name] = overall_query_level_counts.get(name, 0) + int(count)

        cat_columns = infer_categorical_columns(x_train_fit)
        x_train_sorted, y_train_sorted, train_group_sorted = _sort_grouped_frame(
            x_train_fit,
            y_train_fit,
            train_query_ids,
        )
        x_valid_sorted, y_valid_sorted, valid_group_sorted = _sort_grouped_frame(
            x_valid_fit,
            y_valid_fit,
            valid_query_ids,
        )

        train_pairs = _build_sampled_pairs(
            y_train_sorted,
            train_group_sorted,
            max_pairs_per_group=max_pairs_per_group,
            random_seed=random_state + fold_number,
        )
        valid_pairs = _build_sampled_pairs(
            y_valid_sorted,
            valid_group_sorted,
            max_pairs_per_group=max_pairs_per_group,
            random_seed=random_state + folds + fold_number,
        )
        train_pool = Pool(
            x_train_sorted,
            label=y_train_sorted.astype(int),
            group_id=train_group_sorted,
            pairs=train_pairs,
            cat_features=cat_columns,
        )
        valid_pool = Pool(
            x_valid_sorted,
            label=y_valid_sorted.astype(int),
            group_id=valid_group_sorted,
            pairs=valid_pairs,
            cat_features=cat_columns,
        )

        model = _build_ranker(params=params, loss_function=loss_function)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=int(early_stopping_rounds),
            verbose=int(verbose),
        )
        valid_rank_score = pd.Series(model.predict(x_valid_fit), index=x_valid_fit.index, dtype="float64")
        if valid_local_mask is not None:
            ranker_oof.iloc[np.asarray(valid_idx)[valid_local_mask]] = valid_rank_score.values
        else:
            ranker_oof.iloc[valid_idx] = valid_rank_score.values

        fold_best_iteration = _best_iteration_or_default(model, params.iterations)
        fold_iterations.append(int(fold_best_iteration))
        best_iteration_raw = model.get_best_iteration()
        fold_rows.append(
            {
                "fold": int(fold_number),
                "train_rows": int(len(y_train_fit)),
                "valid_rows": int(len(y_valid_fit)),
                "train_query_count": int(len(np.unique(train_query_ids))),
                "valid_query_count": int(len(np.unique(valid_query_ids))),
                "train_query_level_counts": train_level_counts,
                "valid_query_level_counts": valid_level_counts,
                "reference_auc": float(binary_auc(y_valid_fit, valid_reference_eval)),
                "ranker_auc": float(binary_auc(y_valid_fit, valid_rank_score)),
                "train_query_global_share": float(train_levels.eq("global").mean()),
                "valid_query_global_share": float(valid_levels.eq("global").mean()),
                "best_iteration": int(best_iteration_raw) if best_iteration_raw is not None and best_iteration_raw >= 0 else -1,
                "final_iterations": int(fold_best_iteration),
            }
        )

    scoring_mask = local_mask if local_mask is not None else pd.Series(True, index=x_base.index, dtype=bool)
    if ranker_oof.loc[scoring_mask].isna().any():
        raise ValueError("OOF ranker predictions contain NaN values on the scoring mask")
    reference_auc_on_mask = binary_auc(y.loc[scoring_mask], aligned_reference.loc[scoring_mask])
    ranker_auc_on_mask = binary_auc(y.loc[scoring_mask], ranker_oof.loc[scoring_mask])

    alpha_rows: list[dict[str, Any]] = []
    best_alpha = float(alpha_grid[0])
    best_auc = float("-inf")
    best_candidate = aligned_reference.copy()
    for alpha in alpha_grid:
        candidate_score = aligned_reference.copy()
        reordered_reference = _reorder_reference_by_rank(
            aligned_reference.loc[scoring_mask],
            ranker_oof.loc[scoring_mask],
        )
        candidate_score.loc[scoring_mask] = (
            (1.0 - float(alpha)) * aligned_reference.loc[scoring_mask].astype("float64")
            + float(alpha) * reordered_reference
        ).values
        candidate_auc = binary_auc(y, candidate_score)
        candidate_auc_on_mask = binary_auc(y.loc[scoring_mask], candidate_score.loc[scoring_mask])
        alpha_rows.append(
            {
                "alpha": float(alpha),
                "candidate_oof_auc": float(candidate_auc),
                "candidate_oof_auc_on_mask": float(candidate_auc_on_mask),
                "reference_oof_auc": float(binary_auc(y, aligned_reference)),
                "reference_oof_auc_on_mask": float(reference_auc_on_mask),
                "ranker_oof_auc_on_mask": float(ranker_auc_on_mask),
            }
        )
        if candidate_auc > best_auc:
            best_auc = float(candidate_auc)
            best_alpha = float(alpha)
            best_candidate = candidate_score

    x_full = _transform_single_with_stateful_blocks(x_base, stateful_blocks)
    x_full, _ = _append_reference_features(
        x_full,
        reference_pred=aligned_reference,
        reference_component_frame=component_frame,
        include_logit=include_logit_reference,
    )
    if local_mask is not None:
        x_full_fit = x_full.loc[local_mask]
        y_full_fit = y.loc[local_mask]
        full_query_frame = query_candidates.loc[local_mask]
    else:
        x_full_fit = x_full
        y_full_fit = y
        full_query_frame = query_candidates
    full_query_ids, full_query_levels, full_query_level_counts = _assign_query_groups(
        full_query_frame,
        y_full_fit,
        min_rows=min_query_rows,
        min_positive_rows=min_query_positive_rows,
        min_negative_rows=min_query_negative_rows,
    )
    full_cat_columns = infer_categorical_columns(x_full_fit)
    x_full_sorted, y_full_sorted, full_group_sorted = _sort_grouped_frame(x_full_fit, y_full_fit, full_query_ids)
    full_pairs = _build_sampled_pairs(
        y_full_sorted,
        full_group_sorted,
        max_pairs_per_group=max_pairs_per_group,
        random_seed=random_state + 10_000,
    )
    full_pool = Pool(
        x_full_sorted,
        label=y_full_sorted.astype(int),
        group_id=full_group_sorted,
        pairs=full_pairs,
        cat_features=full_cat_columns,
    )
    final_iterations = max(int(np.median(fold_iterations)), 1)
    full_params = CatBoostHyperParams(
        iterations=final_iterations,
        learning_rate=params.learning_rate,
        depth=params.depth,
        l2_leaf_reg=params.l2_leaf_reg,
        random_seed=params.random_seed,
        loss_function=params.loss_function,
        eval_metric=params.eval_metric,
    )
    full_model = _build_ranker(params=full_params, loss_function=loss_function)
    full_model.fit(full_pool, verbose=int(verbose))
    _save_model(full_model, model_path)

    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "is_scoring_mask": scoring_mask.astype(int).values,
            "reference_pred": aligned_reference.values,
            "ranker_score": ranker_oof.values,
            "candidate_score": best_candidate.values,
        }
    )
    ensure_parent_dir(oof_path)
    oof_frame.to_csv(oof_path, index=False)

    metrics: dict[str, Any] = {
        "loss_function": str(loss_function),
        "preset": str(preset) if preset is not None else None,
        "feature_blocks": list(normalized_blocks),
        "feature_count": int(x_full_fit.shape[1]),
        "categorical_columns": full_cat_columns,
        "teacher_disagreement_columns": list(disagreement_columns),
        "cv_folds": int(folds),
        "cv_fold_metrics": fold_rows,
        "fold_final_iterations": [int(value) for value in fold_iterations],
        "final_iterations": int(final_iterations),
        "stratify_mode": normalized_stratify_mode,
        "min_query_rows": int(min_query_rows),
        "min_query_positive_rows": int(min_query_positive_rows),
        "min_query_negative_rows": int(min_query_negative_rows),
        "max_pairs_per_group": int(max_pairs_per_group),
        "query_level_counts_cv_train": overall_query_level_counts,
        "query_level_counts_full_train": full_query_level_counts,
        "full_train_query_count": int(len(np.unique(full_query_ids))),
        "full_train_global_share": float(full_query_levels.eq("global").mean()),
        "scoring_mask_rows": int(scoring_mask.sum()),
        "scoring_mask_share": float(scoring_mask.mean()),
        "reference_oof_auc": float(binary_auc(y, aligned_reference)),
        "reference_oof_auc_on_mask": float(reference_auc_on_mask),
        "ranker_oof_auc_on_mask": float(ranker_auc_on_mask),
        "candidate_oof_auc": float(best_auc),
        "candidate_oof_auc_on_mask": float(binary_auc(y.loc[scoring_mask], best_candidate.loc[scoring_mask])),
        "delta_vs_reference_oof_auc": float(best_auc - binary_auc(y, aligned_reference)),
        "best_alpha": float(best_alpha),
        "alpha_scan": alpha_rows,
        "model_path": str(model_path),
        "oof_path": str(oof_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
    }
    write_json(metrics_path, metrics)
    return metrics
