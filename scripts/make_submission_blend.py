#!/usr/bin/env python3
"""Generate submission CSV from CatBoost+LightGBM+XGBoost weighted blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import ID_COLUMN, TARGET_COLUMN
from churn_baseline.data import load_csv, prepare_test_features, prepare_train_features
from churn_baseline.encoded_features import build_dense_encoded_matrix
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.modeling import load_model, predict_proba


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def _parse_model_paths(raw: str) -> list[str]:
    paths = [token.strip() for token in raw.split(",") if token.strip()]
    if not paths:
        raise ValueError("At least one model path is required")
    return paths


def _model_paths_from_metrics(metrics_path: str | Path) -> list[str]:
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    model_paths = payload.get("model_paths", [])
    if not model_paths:
        raise ValueError(f"No model_paths found in metrics file: {metrics_path}")
    return [str(path) for path in model_paths]


def _load_blend_weights(weights_json_path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(weights_json_path).read_text(encoding="utf-8"))
    weights = payload.get("selected_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(
            f"selected_weights not found in blend JSON: {weights_json_path}. "
            "Run scripts/blend_oof.py first."
        )
    normalized = {str(k): float(v) for k, v in weights.items()}
    total = float(sum(max(v, 0.0) for v in normalized.values()))
    if total <= 0.0:
        raise ValueError("Blend weights sum to 0 after non-negative projection.")
    return {key: max(value, 0.0) / total for key, value in normalized.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate blended submission from diverse models")
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Path to train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--feature-blocks",
        default="none",
        help="Optional feature blocks (A,B,C,O,P) or none.",
    )
    parser.add_argument(
        "--catboost-model-paths",
        default="",
        help="Optional comma-separated CatBoost model paths. If empty, reads --catboost-metrics-path.",
    )
    parser.add_argument(
        "--catboost-metrics-path",
        default="artifacts/reports/train_cv_multiseed_full_hiiter_metrics.json",
        help="Metrics JSON with CatBoost model_paths.",
    )
    parser.add_argument(
        "--lightgbm-model-path",
        default="artifacts/models/lightgbm_cv_full_hiiter.txt",
        help="Path to LightGBM full model.",
    )
    parser.add_argument(
        "--xgboost-model-path",
        default="artifacts/models/xgboost_cv_full_hiiter.json",
        help="Path to XGBoost full model.",
    )
    parser.add_argument(
        "--weights-json-path",
        default="artifacts/reports/model_diversity_blend_3model_full_grid002.json",
        help="Blend JSON from scripts/blend_oof.py (selected_weights).",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/submissions/playground-series-s6e3.csv",
        help="Output submission CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_blocks = _parse_feature_blocks(args.feature_blocks)
    weights = _load_blend_weights(args.weights_json_path)

    if args.catboost_model_paths.strip():
        catboost_model_paths = _parse_model_paths(args.catboost_model_paths)
    else:
        catboost_model_paths = _model_paths_from_metrics(args.catboost_metrics_path)

    train_df = load_csv(args.train_csv)
    test_df = load_csv(args.test_csv)
    if ID_COLUMN not in test_df.columns:
        raise ValueError(f"Column '{ID_COLUMN}' not found in test CSV")

    ids = test_df[ID_COLUMN].copy()
    x_train_raw, _ = prepare_train_features(train_df, drop_id=True, feature_blocks=feature_blocks)
    x_test_raw = prepare_test_features(test_df, drop_id=True, feature_blocks=feature_blocks)

    cb_preds = []
    for model_path in catboost_model_paths:
        model = load_model(model_path)
        cb_preds.append(predict_proba(model, x_test_raw).values)
    pred_cb = np.mean(np.column_stack(cb_preds), axis=1)

    train_encoded = build_dense_encoded_matrix(x_train_raw)
    test_encoded = pd.get_dummies(x_test_raw, dummy_na=False)
    x_test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0.0)
    x_test_encoded = x_test_encoded.astype("float32")

    lgb_model = lgb.Booster(model_file=str(args.lightgbm_model_path))
    pred_lgb = lgb_model.predict(x_test_encoded)

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(args.xgboost_model_path))
    pred_xgb = xgb_model.predict_proba(x_test_encoded)[:, 1]

    available = {
        "pred_cb": pred_cb,
        "pred_lgb": pred_lgb,
        "pred_xgb": pred_xgb,
    }
    missing = [key for key in weights.keys() if key not in available]
    if missing:
        raise ValueError(f"Weights contain unknown prediction keys: {missing}")

    final_pred = np.zeros(shape=len(test_df), dtype="float64")
    for key, weight in weights.items():
        final_pred += float(weight) * available[key]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission = test_df[[ID_COLUMN]].copy()
    submission[TARGET_COLUMN] = final_pred
    submission.to_csv(out_path, index=False)

    summary = {
        "output_csv": str(out_path),
        "rows": int(len(submission)),
        "feature_blocks": feature_blocks,
        "catboost_model_count": int(len(catboost_model_paths)),
        "lightgbm_model_path": str(args.lightgbm_model_path),
        "xgboost_model_path": str(args.xgboost_model_path),
        "weights_json_path": str(args.weights_json_path),
        "weights": weights,
        "prediction_min": float(final_pred.min()),
        "prediction_max": float(final_pred.max()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
