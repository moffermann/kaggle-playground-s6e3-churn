"""Minimal transductive GraphSAGE probe for the churn competition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pynndescent import NNDescent
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.nn import SAGEConv

from .artifacts import ensure_parent_dir, write_json
from .config import ID_COLUMN, TARGET_COLUMN
from .data import encode_target, load_csv
from .evaluation import binary_auc


BASE_NUMERIC_COLUMNS: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")
BASE_CATEGORICAL_COLUMNS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class GNNProbeParams:
    """Hyperparameters for the minimal GraphSAGE smoke."""

    hidden_dim: int = 32
    dropout: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 3e-4
    epochs: int = 8
    patience: int = 3
    k_neighbors: int = 8
    graph_numeric_multiplier: float = 3.0
    random_state: int = 42


def _set_reproducible_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _clip_probability(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    return np.clip(values.astype("float64"), epsilon, 1.0 - epsilon)


def _clean_total_charges(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    return out


def _prepare_base_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _clean_total_charges(train_df.copy())
    test = _clean_total_charges(test_df.copy())

    for column in BASE_NUMERIC_COLUMNS:
        train[column] = pd.to_numeric(train[column], errors="coerce")
        test[column] = pd.to_numeric(test[column], errors="coerce")

    for column in BASE_CATEGORICAL_COLUMNS:
        train[column] = train[column].fillna("missing").astype(str).str.strip()
        test[column] = test[column].fillna("missing").astype(str).str.strip()

    return train, test


def _encode_categorical_node_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    train_codes: list[np.ndarray] = []
    test_codes: list[np.ndarray] = []
    cardinalities: list[int] = []

    for column in BASE_CATEGORICAL_COLUMNS:
        combined = pd.concat(
            [train_frame[column].astype(str), test_frame[column].astype(str)],
            axis=0,
            ignore_index=True,
        )
        unique_values = combined.unique().tolist()
        mapping = {value: index for index, value in enumerate(unique_values)}
        train_codes.append(train_frame[column].astype(str).map(mapping).astype(np.int64).to_numpy())
        test_codes.append(test_frame[column].astype(str).map(mapping).astype(np.int64).to_numpy())
        cardinalities.append(int(len(mapping)))

    x_cat_train = np.stack(train_codes, axis=1)
    x_cat_test = np.stack(test_codes, axis=1)
    return x_cat_train, x_cat_test, cardinalities


def _build_graph_matrix(
    fit_frame: pd.DataFrame,
    query_frames: Sequence[pd.DataFrame],
    *,
    graph_numeric_multiplier: float,
) -> tuple[np.ndarray, list[np.ndarray], StandardScaler]:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
    fit_cat = fit_frame.loc[:, list(BASE_CATEGORICAL_COLUMNS)]
    x_cat_fit = ohe.fit_transform(fit_cat).astype(np.float32)
    x_cat_queries = [
        ohe.transform(frame.loc[:, list(BASE_CATEGORICAL_COLUMNS)]).astype(np.float32) for frame in query_frames
    ]

    scaler = StandardScaler()
    x_num_fit = scaler.fit_transform(fit_frame.loc[:, list(BASE_NUMERIC_COLUMNS)].to_numpy(dtype=np.float32)).astype(
        np.float32
    )
    x_num_fit *= float(graph_numeric_multiplier)
    x_num_queries = [
        scaler.transform(frame.loc[:, list(BASE_NUMERIC_COLUMNS)].to_numpy(dtype=np.float32)).astype(np.float32)
        * float(graph_numeric_multiplier)
        for frame in query_frames
    ]

    fit_matrix = np.concatenate([x_cat_fit, x_num_fit], axis=1).astype(np.float32)
    query_matrices = [
        np.concatenate([cat_block, num_block], axis=1).astype(np.float32)
        for cat_block, num_block in zip(x_cat_queries, x_num_queries, strict=True)
    ]
    return fit_matrix, query_matrices, scaler


def _build_neighbor_graph(
    fit_graph_matrix: np.ndarray,
    query_graph_matrices: Sequence[np.ndarray],
    *,
    k_neighbors: int,
    random_state: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    index = NNDescent(
        fit_graph_matrix,
        n_neighbors=int(k_neighbors) + 1,
        metric="euclidean",
        random_state=int(random_state),
        n_jobs=-1,
    )
    neighbor_indices, _ = index.neighbor_graph
    fit_size = fit_graph_matrix.shape[0]
    cleaned_fit = np.empty((fit_size, int(k_neighbors)), dtype=np.int64)

    for node in range(fit_size):
        row = [int(value) for value in neighbor_indices[node] if int(value) != node]
        if len(row) < int(k_neighbors):
            row.extend([node] * (int(k_neighbors) - len(row)))
        cleaned_fit[node] = np.asarray(row[: int(k_neighbors)], dtype=np.int64)

    query_neighbors: list[np.ndarray] = []
    for matrix in query_graph_matrices:
        indices, _ = index.query(matrix, k=int(k_neighbors))
        query_neighbors.append(indices.astype(np.int64))
    return cleaned_fit, query_neighbors


def _build_edge_index(
    fit_neighbors: np.ndarray,
    valid_neighbors: np.ndarray,
    test_neighbors: np.ndarray,
    *,
    fit_size: int,
    valid_size: int,
    device: torch.device,
) -> torch.Tensor:
    fit_targets = np.repeat(np.arange(fit_size, dtype=np.int64), fit_neighbors.shape[1])
    valid_targets = np.repeat(np.arange(fit_size, fit_size + valid_size, dtype=np.int64), valid_neighbors.shape[1])
    test_start = fit_size + valid_size
    test_targets = np.repeat(np.arange(test_start, test_start + test_neighbors.shape[0], dtype=np.int64), test_neighbors.shape[1])

    source = np.concatenate(
        [
            fit_neighbors.reshape(-1).astype(np.int64),
            valid_neighbors.reshape(-1).astype(np.int64),
            test_neighbors.reshape(-1).astype(np.int64),
        ],
        axis=0,
    )
    destination = np.concatenate([fit_targets, valid_targets, test_targets], axis=0)
    edge_index = np.vstack([source, destination])
    return torch.tensor(edge_index, dtype=torch.long, device=device)


def _resolve_device(device: str) -> torch.device:
    normalized = str(device).strip().lower()
    if normalized in {"auto", ""}:
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        normalized = "cpu"
    return torch.device(normalized)


def _embedding_dim(cardinality: int) -> int:
    return int(min(12, max(2, round(cardinality ** 0.5))))


class GraphSageWithCategoricals(nn.Module):
    """Small GraphSAGE with categorical embeddings and numeric inputs."""

    def __init__(
        self,
        *,
        numeric_dim: int,
        cardinalities: Sequence[int],
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinality), _embedding_dim(int(cardinality))) for cardinality in cardinalities]
        )
        embedding_dim = sum(embedding.embedding_dim for embedding in self.embeddings)
        self.input_linear = nn.Linear(int(numeric_dim) + int(embedding_dim), int(hidden_dim))
        self.conv1 = SAGEConv(int(hidden_dim), int(hidden_dim))
        self.conv2 = SAGEConv(int(hidden_dim), int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)
        self.dropout = float(dropout)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [embedding(x_cat[:, index]) for index, embedding in enumerate(self.embeddings)]
        x = torch.cat([x_num, *embedded], dim=1)
        x = F.relu(self.input_linear(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x).squeeze(-1)


def _to_analysis_oof(
    *,
    reference_v3_oof_path: str | Path,
    candidate_oof_path: str | Path,
    analysis_oof_path: str | Path,
) -> str:
    reference = pd.read_csv(reference_v3_oof_path)[[ID_COLUMN, "target", "candidate_pred"]].rename(
        columns={"candidate_pred": "reference_pred"}
    )
    candidate = pd.read_csv(candidate_oof_path)[[ID_COLUMN, "target", "oof_pred"]].rename(
        columns={"oof_pred": "candidate_pred"}
    )
    merged = reference.merge(candidate, on=[ID_COLUMN, "target"], how="inner", validate="one_to_one")
    out_path = ensure_parent_dir(analysis_oof_path)
    merged.to_csv(out_path, index=False)
    return str(out_path)


def train_gnn_probe_cv(
    *,
    train_csv_path: str | Path,
    test_csv_path: str | Path,
    metrics_path: str | Path,
    oof_path: str | Path,
    test_pred_path: str | Path | None = None,
    analysis_oof_path: str | Path | None = None,
    reference_v3_oof_path: str | Path | None = None,
    params: GNNProbeParams | None = None,
    folds: int = 2,
    random_state: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """Train the minimal transductive GraphSAGE probe with outer CV."""
    params = params or GNNProbeParams(random_state=random_state)
    _set_reproducible_seed(int(params.random_state))

    train_df = load_csv(train_csv_path)
    test_df = load_csv(test_csv_path)
    train_df, test_df = _prepare_base_frames(train_df, test_df)
    y = encode_target(train_df[TARGET_COLUMN]).astype("float32")

    train_ids = train_df[ID_COLUMN].astype("int64").reset_index(drop=True)
    test_ids = test_df[ID_COLUMN].astype("int64").reset_index(drop=True)

    train_features = train_df.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
    test_features = test_df.reset_index(drop=True)
    n_train = int(len(train_features))
    model_device = _resolve_device(device)
    x_cat_train_all, x_cat_test_all, cardinalities = _encode_categorical_node_features(train_features, test_features)

    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(random_state))
    oof_pred = np.zeros(n_train, dtype=np.float32)
    test_pred = np.zeros(len(test_features), dtype=np.float32)
    fold_aucs: list[float] = []
    best_epochs: list[int] = []
    graph_feature_dim: int | None = None

    for fold_idx, (fit_idx, valid_idx) in enumerate(splitter.split(np.zeros(n_train), y.to_numpy(dtype=np.int8)), start=1):
        _set_reproducible_seed(int(params.random_state) + fold_idx)
        fit_frame = train_features.iloc[fit_idx].reset_index(drop=True)
        valid_frame = train_features.iloc[valid_idx].reset_index(drop=True)
        test_frame = test_features.reset_index(drop=True)

        fill_values = fit_frame.loc[:, list(BASE_NUMERIC_COLUMNS)].median(numeric_only=True)
        for column in BASE_NUMERIC_COLUMNS:
            fill_value = float(fill_values.get(column, 0.0))
            fit_frame[column] = fit_frame[column].fillna(fill_value).astype(np.float32)
            valid_frame[column] = valid_frame[column].fillna(fill_value).astype(np.float32)
            test_frame[column] = test_frame[column].fillna(fill_value).astype(np.float32)

        fit_graph, query_graphs, scaler = _build_graph_matrix(
            fit_frame,
            [valid_frame, test_frame],
            graph_numeric_multiplier=params.graph_numeric_multiplier,
        )
        if graph_feature_dim is None:
            graph_feature_dim = int(fit_graph.shape[1])
        fit_neighbors, query_neighbors = _build_neighbor_graph(
            fit_graph,
            query_graphs,
            k_neighbors=params.k_neighbors,
            random_state=params.random_state + fold_idx,
        )
        valid_neighbors, test_neighbors = query_neighbors

        x_num_fit = scaler.transform(fit_frame.loc[:, list(BASE_NUMERIC_COLUMNS)].to_numpy(dtype=np.float32)).astype(np.float32)
        x_num_valid = scaler.transform(valid_frame.loc[:, list(BASE_NUMERIC_COLUMNS)].to_numpy(dtype=np.float32)).astype(np.float32)
        x_num_test = scaler.transform(test_frame.loc[:, list(BASE_NUMERIC_COLUMNS)].to_numpy(dtype=np.float32)).astype(np.float32)

        x_cat_fit = x_cat_train_all[fit_idx]
        x_cat_valid = x_cat_train_all[valid_idx]
        x_cat_test = x_cat_test_all

        all_num = np.vstack([x_num_fit, x_num_valid, x_num_test]).astype(np.float32)
        all_cat = np.vstack([x_cat_fit, x_cat_valid, x_cat_test]).astype(np.int64)
        fit_size = int(len(fit_idx))
        valid_size = int(len(valid_idx))
        total_nodes = int(all_num.shape[0])
        valid_local_index = torch.arange(fit_size, fit_size + valid_size, dtype=torch.long, device=model_device)
        fit_local_index = torch.arange(0, fit_size, dtype=torch.long, device=model_device)
        test_local_index = torch.arange(fit_size + valid_size, total_nodes, dtype=torch.long, device=model_device)

        x_num_tensor = torch.tensor(all_num, dtype=torch.float32, device=model_device)
        x_cat_tensor = torch.tensor(all_cat, dtype=torch.long, device=model_device)
        y_tensor = torch.tensor(y.iloc[fit_idx].to_numpy(dtype=np.float32), dtype=torch.float32, device=model_device)
        edge_index = _build_edge_index(
            fit_neighbors,
            valid_neighbors,
            test_neighbors,
            fit_size=fit_size,
            valid_size=valid_size,
            device=model_device,
        )

        model = GraphSageWithCategoricals(
            numeric_dim=all_num.shape[1],
            cardinalities=cardinalities,
            hidden_dim=params.hidden_dim,
            dropout=params.dropout,
        ).to(model_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc = -np.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0

        for epoch in range(1, int(params.epochs) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_num_tensor, x_cat_tensor, edge_index)
            loss = loss_fn(logits[fit_local_index], y_tensor)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(x_num_tensor, x_cat_tensor, edge_index)
                valid_prob = torch.sigmoid(logits[valid_local_index]).detach().cpu().numpy()
            valid_auc = float(binary_auc(y.iloc[valid_idx], valid_prob))

            if valid_auc > best_auc + 1e-6:
                best_auc = valid_auc
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(params.patience):
                    break

        if best_state is None:
            raise RuntimeError(f"Fold {fold_idx} failed to produce a best state.")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(x_num_tensor, x_cat_tensor, edge_index)
            valid_prob = torch.sigmoid(logits[valid_local_index]).detach().cpu().numpy().astype(np.float32)
            test_prob = torch.sigmoid(logits[test_local_index]).detach().cpu().numpy().astype(np.float32)

        oof_pred[valid_idx] = valid_prob
        test_pred += test_prob / float(folds)
        fold_aucs.append(float(binary_auc(y.iloc[valid_idx], valid_prob)))
        best_epochs.append(int(best_epoch))

        del model, optimizer, fit_local_index, valid_local_index, test_local_index
        if model_device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = {
        "experiment_name": "gnn_probe",
        "model_family": "gnn_graphsage",
        "train_csv_path": str(train_csv_path),
        "test_csv_path": str(test_csv_path),
        "folds": int(folds),
        "random_state": int(random_state),
        "device": str(model_device),
        "oof_auc": float(binary_auc(y, oof_pred)),
        "cv_mean_auc": float(np.mean(fold_aucs)),
        "cv_std_auc": float(np.std(fold_aucs, ddof=0)),
        "fold_aucs": [float(value) for value in fold_aucs],
        "best_epochs": [int(value) for value in best_epochs],
        "train_rows": int(n_train),
        "test_rows": int(len(test_features)),
        "total_nodes": int(n_train + len(test_features)),
        "graph_feature_dim": int(graph_feature_dim or 0),
        "edge_count_per_fold": int(params.k_neighbors * (n_train + len(test_features))),
        "categorical_columns": list(BASE_CATEGORICAL_COLUMNS),
        "numeric_columns": list(BASE_NUMERIC_COLUMNS),
        "categorical_cardinalities": [int(value) for value in cardinalities],
        "params": asdict(params),
        "oof_path": str(oof_path),
        "metrics_path": str(metrics_path),
    }

    oof_frame = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "target": y.astype("int8"),
            "oof_pred": _clip_probability(oof_pred),
        }
    )
    oof_out = ensure_parent_dir(oof_path)
    oof_frame.to_csv(oof_out, index=False)

    if test_pred_path:
        test_out = ensure_parent_dir(test_pred_path)
        pd.DataFrame(
            {
                ID_COLUMN: test_ids,
                TARGET_COLUMN: _clip_probability(test_pred),
            }
        ).to_csv(test_out, index=False)
        metrics["test_pred_path"] = str(test_pred_path)

    if analysis_oof_path and reference_v3_oof_path:
        metrics["analysis_oof_path"] = _to_analysis_oof(
            reference_v3_oof_path=reference_v3_oof_path,
            candidate_oof_path=oof_path,
            analysis_oof_path=analysis_oof_path,
        )

    write_json(metrics_path, metrics)
    return metrics
