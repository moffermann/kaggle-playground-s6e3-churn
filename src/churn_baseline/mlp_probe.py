"""PyTorch tabular MLP probe with categorical embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import infer_categorical_columns, load_csv, prepare_train_features
from .evaluation import binary_auc
from .feature_engineering import normalize_feature_blocks


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(device: str) -> torch.device:
    device_value = str(device).strip().lower()
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def _parse_hidden_dims(raw: Sequence[int] | str) -> list[int]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return [int(part) for part in parts]
    return [int(value) for value in raw]


def _embedding_dim(cardinality: int, max_dim: int) -> int:
    return max(4, min(int(max_dim), int(round(1.6 * (int(cardinality) ** 0.56)))))


def _fit_category_maps(frame: pd.DataFrame, categorical_columns: Sequence[str]) -> tuple[dict[str, dict[str, int]], list[int]]:
    mappings: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []
    for column in categorical_columns:
        values = frame[column].fillna("__MISSING__").astype(str)
        unique_values = sorted(values.unique().tolist())
        mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
        mappings[str(column)] = mapping
        cardinalities.append(len(mapping) + 1)
    return mappings, cardinalities


def _encode_categorical(
    frame: pd.DataFrame,
    categorical_columns: Sequence[str],
    mappings: dict[str, dict[str, int]],
) -> np.ndarray:
    if not categorical_columns:
        return np.zeros((len(frame), 0), dtype=np.int64)

    encoded = np.zeros((len(frame), len(categorical_columns)), dtype=np.int64)
    for col_idx, column in enumerate(categorical_columns):
        series = frame[column].fillna("__MISSING__").astype(str)
        mapping = mappings[str(column)]
        encoded[:, col_idx] = series.map(mapping).fillna(0).astype(np.int64).to_numpy()
    return encoded


def _fit_numeric_stats(frame: pd.DataFrame, numeric_columns: Sequence[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    if not numeric_columns:
        empty = pd.Series(dtype="float64")
        return empty, empty, empty
    subset = frame.loc[:, list(numeric_columns)].copy()
    medians = subset.median()
    means = subset.fillna(medians).mean()
    stds = subset.fillna(medians).std(ddof=0).replace(0.0, 1.0)
    return medians, means, stds


def _encode_numeric(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    medians: pd.Series,
    means: pd.Series,
    stds: pd.Series,
) -> np.ndarray:
    if not numeric_columns:
        return np.zeros((len(frame), 0), dtype=np.float32)
    subset = frame.loc[:, list(numeric_columns)].copy()
    subset = subset.fillna(medians)
    subset = (subset - means) / stds
    subset = subset.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return subset.astype("float32").to_numpy()


class TabularEmbeddingMLP(nn.Module):
    def __init__(
        self,
        *,
        cardinalities: Sequence[int],
        numeric_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        max_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinality), _embedding_dim(int(cardinality), max_embedding_dim)) for cardinality in cardinalities]
        )
        input_dim = sum(embedding.embedding_dim for embedding in self.embeddings) + int(numeric_dim)
        layers: list[nn.Module] = []
        hidden_dims_list = [int(value) for value in hidden_dims if int(value) > 0]
        for hidden_dim in hidden_dims_list:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(float(dropout)),
                ]
            )
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.embeddings:
            embedded = [embedding(x_cat[:, index]) for index, embedding in enumerate(self.embeddings)]
            parts.append(torch.cat(embedded, dim=1))
        if x_num.shape[1] > 0:
            parts.append(x_num)
        if not parts:
            raise ValueError("MLP probe requires at least one categorical or numeric feature.")
        combined = torch.cat(parts, dim=1)
        hidden = self.backbone(combined)
        return self.output(hidden).squeeze(1)


@dataclass(frozen=True)
class MLPProbeParams:
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    epochs: int = 6
    max_embedding_dim: int = 32
    device: str = "auto"


def _build_dataset(
    x_cat: np.ndarray,
    x_num: np.ndarray,
    y: np.ndarray | None = None,
) -> TensorDataset:
    cat_tensor = torch.from_numpy(x_cat.astype(np.int64))
    num_tensor = torch.from_numpy(x_num.astype(np.float32))
    if y is None:
        target_tensor = torch.zeros((len(x_cat),), dtype=torch.float32)
    else:
        target_tensor = torch.from_numpy(y.astype(np.float32))
    return TensorDataset(cat_tensor, num_tensor, target_tensor)


def _train_one_fold(
    *,
    x_train_cat: np.ndarray,
    x_train_num: np.ndarray,
    y_train: np.ndarray,
    x_valid_cat: np.ndarray,
    x_valid_num: np.ndarray,
    y_valid: np.ndarray,
    cardinalities: Sequence[int],
    params: MLPProbeParams,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    _set_torch_seed(seed)
    model = TabularEmbeddingMLP(
        cardinalities=cardinalities,
        numeric_dim=x_train_num.shape[1],
        hidden_dims=params.hidden_dims,
        dropout=params.dropout,
        max_embedding_dim=params.max_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.learning_rate),
        weight_decay=float(params.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    dataset = _build_dataset(x_train_cat, x_train_num, y_train)
    loader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    x_valid_cat_tensor = torch.from_numpy(x_valid_cat.astype(np.int64)).to(device)
    x_valid_num_tensor = torch.from_numpy(x_valid_num.astype(np.float32)).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -1.0
    best_epoch = 1

    for epoch in range(1, int(params.epochs) + 1):
        model.train()
        for batch_cat, batch_num, batch_target in loader:
            batch_cat = batch_cat.to(device, non_blocking=device.type == "cuda")
            batch_num = batch_num.to(device, non_blocking=device.type == "cuda")
            batch_target = batch_target.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits = model(x_valid_cat_tensor, x_valid_num_tensor)
            valid_pred = torch.sigmoid(valid_logits).detach().cpu().numpy()
        valid_auc = float(binary_auc(pd.Series(y_valid), pd.Series(valid_pred)))
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("MLP probe did not produce a valid best_state.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred = torch.sigmoid(model(x_valid_cat_tensor, x_valid_num_tensor)).detach().cpu().numpy()
    return final_pred.astype(np.float64), best_epoch


def _fit_full_model(
    *,
    x_cat: np.ndarray,
    x_num: np.ndarray,
    y: np.ndarray,
    cardinalities: Sequence[int],
    params: MLPProbeParams,
    final_epochs: int,
    seed: int,
    device: torch.device,
) -> TabularEmbeddingMLP:
    _set_torch_seed(seed)
    model = TabularEmbeddingMLP(
        cardinalities=cardinalities,
        numeric_dim=x_num.shape[1],
        hidden_dims=params.hidden_dims,
        dropout=params.dropout,
        max_embedding_dim=params.max_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.learning_rate),
        weight_decay=float(params.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    dataset = _build_dataset(x_cat, x_num, y)
    loader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    for _ in range(int(final_epochs)):
        model.train()
        for batch_cat, batch_num, batch_target in loader:
            batch_cat = batch_cat.to(device, non_blocking=device.type == "cuda")
            batch_num = batch_num.to(device, non_blocking=device.type == "cuda")
            batch_target = batch_target.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_target)
            loss.backward()
            optimizer.step()
    return model.cpu()


def run_mlp_probe_cv(
    *,
    train_csv_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
    feature_blocks: Sequence[str] | None,
    folds: int,
    random_state: int,
    params: MLPProbeParams,
    reference_pred: pd.Series | None = None,
    alpha_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Train a shallow tabular MLP with categorical embeddings."""
    if folds < 2:
        raise ValueError("folds must be >= 2")

    train_df = load_csv(train_csv_path)
    normalized_blocks = normalize_feature_blocks(feature_blocks)
    x, y = prepare_train_features(train_df, drop_id=True, feature_blocks=normalized_blocks)
    categorical_columns = infer_categorical_columns(x)
    numeric_columns = [column for column in x.columns if column not in categorical_columns]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    device = _resolve_device(params.device)

    oof_pred = pd.Series(index=x.index, dtype="float64", name="oof_pred")
    fold_rows: list[dict[str, Any]] = []
    best_epochs: list[int] = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        mappings, cardinalities = _fit_category_maps(x_train, categorical_columns)
        medians, means, stds = _fit_numeric_stats(x_train, numeric_columns)
        x_train_cat = _encode_categorical(x_train, categorical_columns, mappings)
        x_valid_cat = _encode_categorical(x_valid, categorical_columns, mappings)
        x_train_num = _encode_numeric(x_train, numeric_columns, medians, means, stds)
        x_valid_num = _encode_numeric(x_valid, numeric_columns, medians, means, stds)

        fold_pred, best_epoch = _train_one_fold(
            x_train_cat=x_train_cat,
            x_train_num=x_train_num,
            y_train=y_train.to_numpy(dtype=np.float32),
            x_valid_cat=x_valid_cat,
            x_valid_num=x_valid_num,
            y_valid=y_valid.to_numpy(dtype=np.float32),
            cardinalities=cardinalities,
            params=params,
            seed=random_state + fold_number,
            device=device,
        )
        best_epochs.append(int(best_epoch))
        fold_series = pd.Series(fold_pred, index=x_valid.index, dtype="float64")
        oof_pred.iloc[valid_idx] = fold_series.values
        fold_rows.append(
            {
                "fold": int(fold_number),
                "valid_rows": int(len(valid_idx)),
                "auc": float(binary_auc(y_valid, fold_series)),
                "best_epoch": int(best_epoch),
            }
        )

    if oof_pred.isna().any():
        raise RuntimeError("OOF predictions contain missing values for MLP probe.")

    final_epochs = max(int(round(float(np.mean(best_epochs)))), 1)
    full_mappings, full_cardinalities = _fit_category_maps(x, categorical_columns)
    full_medians, full_means, full_stds = _fit_numeric_stats(x, numeric_columns)
    full_x_cat = _encode_categorical(x, categorical_columns, full_mappings)
    full_x_num = _encode_numeric(x, numeric_columns, full_medians, full_means, full_stds)
    full_model = _fit_full_model(
        x_cat=full_x_cat,
        x_num=full_x_num,
        y=y.to_numpy(dtype=np.float32),
        cardinalities=full_cardinalities,
        params=params,
        final_epochs=final_epochs,
        seed=random_state,
        device=device,
    )
    out_model_path = ensure_parent_dir(model_path)
    torch.save(
        {
            "state_dict": full_model.state_dict(),
            "categorical_columns": list(categorical_columns),
            "numeric_columns": list(numeric_columns),
            "category_mappings": full_mappings,
            "cardinalities": list(full_cardinalities),
            "medians": full_medians.to_dict(),
            "means": full_means.to_dict(),
            "stds": full_stds.to_dict(),
            "params": {
                "hidden_dims": list(params.hidden_dims),
                "dropout": float(params.dropout),
                "learning_rate": float(params.learning_rate),
                "weight_decay": float(params.weight_decay),
                "batch_size": int(params.batch_size),
                "epochs": int(params.epochs),
                "max_embedding_dim": int(params.max_embedding_dim),
                "device": str(params.device),
                "final_epochs": int(final_epochs),
            },
        },
        out_model_path,
    )

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_df[ID_COLUMN].values,
            "target": y.astype(int).values,
            "oof_pred": oof_pred.values,
        }
    )
    out_oof_path = ensure_parent_dir(oof_path)
    oof_output.to_csv(out_oof_path, index=False)

    metrics: dict[str, Any] = {
        "train_rows": int(len(train_df)),
        "feature_count": int(x.shape[1]),
        "feature_blocks": list(normalized_blocks),
        "categorical_columns": list(categorical_columns),
        "numeric_columns": list(numeric_columns),
        "cv_folds": int(folds),
        "cv_fold_metrics": fold_rows,
        "cv_mean_auc": float(np.mean([row["auc"] for row in fold_rows])),
        "cv_std_auc": float(np.std([row["auc"] for row in fold_rows])),
        "oof_auc": float(binary_auc(y, oof_pred)),
        "model_path": str(model_path),
        "oof_path": str(oof_path),
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "model_family": "tabular_embedding_mlp",
        "params": {
            "hidden_dims": list(params.hidden_dims),
            "dropout": float(params.dropout),
            "learning_rate": float(params.learning_rate),
            "weight_decay": float(params.weight_decay),
            "batch_size": int(params.batch_size),
            "epochs": int(params.epochs),
            "max_embedding_dim": int(params.max_embedding_dim),
            "device": str(params.device),
            "best_epochs": [int(epoch) for epoch in best_epochs],
            "final_epochs": int(final_epochs),
        },
    }

    if reference_pred is not None:
        reference_by_id = pd.Series(reference_pred, copy=True)
        aligned_reference = train_df[ID_COLUMN].map(reference_by_id)
        if aligned_reference.isna().any():
            missing_ids = train_df.loc[aligned_reference.isna(), ID_COLUMN].head(5).tolist()
            raise ValueError(f"reference_pred missing ids from train: {missing_ids}")
        aligned_reference = pd.Series(aligned_reference.values, index=x.index, dtype="float64")
        metrics["reference_oof_auc"] = float(binary_auc(y, aligned_reference))
        metrics["pearson_corr_vs_reference"] = float(aligned_reference.corr(oof_pred))
        metrics["spearman_corr_vs_reference"] = float(aligned_reference.corr(oof_pred, method="spearman"))

        if alpha_grid:
            alpha_rows: list[dict[str, Any]] = []
            for alpha in alpha_grid:
                alpha_value = float(alpha)
                candidate = (1.0 - alpha_value) * aligned_reference + alpha_value * oof_pred
                alpha_rows.append(
                    {
                        "alpha": alpha_value,
                        "oof_auc": float(binary_auc(y, candidate)),
                    }
                )
            best_alpha_row = max(alpha_rows, key=lambda row: row["oof_auc"])
            metrics["blend_scan"] = alpha_rows
            metrics["best_blend_alpha"] = float(best_alpha_row["alpha"])
            metrics["best_blend_oof_auc"] = float(best_alpha_row["oof_auc"])
            metrics["delta_best_blend_vs_reference"] = float(
                best_alpha_row["oof_auc"] - binary_auc(y, aligned_reference)
            )

    write_json(metrics_path, metrics)
    return metrics
