#!/usr/bin/env python3
"""Run the minimal external Telco transfer-feature smoke against v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import ensure_parent_dir, write_json
from churn_baseline.config import CatBoostHyperParams, ID_COLUMN
from churn_baseline.diagnostics import OOF_TARGET_COLUMN, load_reference_prediction_frame
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.incumbent_v3 import compute_repeated_cv_auc_stats
from churn_baseline.noise_audit import DEFAULT_V3_OOF_SPEC, DOMINANT_MACROFAMILY
from churn_baseline.telco_transfer import DEFAULT_ORIGINAL_CSV, run_telco_transfer_smoke
from churn_baseline.validation_protocol import evaluate_validation_protocol


def _parse_feature_blocks(raw: str) -> list[str]:
    normalized_raw = raw.strip().lower()
    if normalized_raw in {"none", "off", "baseline", ""}:
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return list(normalize_feature_blocks(tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal external Telco transfer-feature smoke against v3")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--original-csv", default=DEFAULT_ORIGINAL_CSV)
    parser.add_argument("--v3-oof", default=DEFAULT_V3_OOF_SPEC)
    parser.add_argument("--feature-blocks", default="R,V")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-mode", choices=("target", "composite"), default="target")
    parser.add_argument("--teacher-valid-size", type=float, default=0.2)
    parser.add_argument("--teacher-iterations", type=int, default=800)
    parser.add_argument("--teacher-learning-rate", type=float, default=0.05)
    parser.add_argument("--teacher-depth", type=int, default=6)
    parser.add_argument("--teacher-l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--teacher-early-stopping-rounds", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--label", default="telco_transfer_smoke")
    parser.add_argument("--out-dir", default="artifacts/reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_blocks = _parse_feature_blocks(args.feature_blocks)
    teacher_params = CatBoostHyperParams(
        iterations=args.teacher_iterations,
        learning_rate=args.teacher_learning_rate,
        depth=args.teacher_depth,
        l2_leaf_reg=args.teacher_l2_leaf_reg,
        random_seed=args.seed,
    )
    challenger_params = CatBoostHyperParams(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.seed,
    )

    metrics = run_telco_transfer_smoke(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        original_csv_path=args.original_csv,
        teacher_params=teacher_params,
        challenger_params=challenger_params,
        feature_blocks=feature_blocks,
        folds=args.folds,
        seed=args.seed,
        stratify_mode=args.stratify_mode,
        teacher_valid_size=args.teacher_valid_size,
        teacher_early_stopping_rounds=args.teacher_early_stopping_rounds,
        challenger_early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
    )

    out_dir = Path(args.out_dir)
    metrics_path = ensure_parent_dir(out_dir / f"{args.label}_metrics.json")
    oof_path = ensure_parent_dir(out_dir / f"{args.label}_oof.csv")
    analysis_oof_path = ensure_parent_dir(out_dir / f"{args.label}_analysis_oof.csv")
    candidate_metrics_path = ensure_parent_dir(out_dir / f"{args.label}_candidate_metrics.json")
    reference_metrics_path = ensure_parent_dir(out_dir / f"{args.label}_reference_v3_metrics.json")
    transfer_train_path = ensure_parent_dir(out_dir / f"{args.label}_transfer_train.csv")
    transfer_test_path = ensure_parent_dir(out_dir / f"{args.label}_transfer_test.csv")
    verdict_path = ensure_parent_dir(out_dir / f"validation_protocol_{args.label}_vs_v3_smoke.json")

    train_frame = pd.read_csv(args.train_csv, usecols=[ID_COLUMN, "Churn"])
    train_frame[OOF_TARGET_COLUMN] = (train_frame["Churn"].astype(str).str.lower() == "yes").astype("int8")
    train_frame = train_frame.drop(columns=["Churn"])
    train_frame["oof_pred"] = metrics["oof_pred"].astype(float).values
    train_frame.to_csv(oof_path, index=False)

    metrics["transfer_train"].to_csv(transfer_train_path, index=False)
    metrics["transfer_test"].to_csv(transfer_test_path, index=False)

    reference_frame, _ = load_reference_prediction_frame(
        reference_oof_spec=args.v3_oof,
        id_column=ID_COLUMN,
        target_column=OOF_TARGET_COLUMN,
    )
    analysis_frame = reference_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "reference_pred"]].merge(
        train_frame[[ID_COLUMN, OOF_TARGET_COLUMN, "oof_pred"]].rename(columns={"oof_pred": "candidate_pred"}),
        how="inner",
        on=[ID_COLUMN, OOF_TARGET_COLUMN],
        validate="one_to_one",
    )
    analysis_frame.to_csv(analysis_oof_path, index=False)

    candidate_metrics = compute_repeated_cv_auc_stats(
        analysis_frame[OOF_TARGET_COLUMN],
        analysis_frame["candidate_pred"],
    )
    reference_metrics = compute_repeated_cv_auc_stats(
        analysis_frame[OOF_TARGET_COLUMN],
        analysis_frame["reference_pred"],
    )
    write_json(candidate_metrics_path, candidate_metrics)
    write_json(reference_metrics_path, reference_metrics)

    verdict = evaluate_validation_protocol(
        train_csv_path=args.train_csv,
        test_csv_path=args.test_csv,
        stage="smoke",
        analysis_oof_path=analysis_oof_path,
        target_family_level="segment3",
        target_family_value=DOMINANT_MACROFAMILY,
        dominant_family_value=DOMINANT_MACROFAMILY,
        candidate_metrics_json=candidate_metrics_path,
        reference_metrics_json=reference_metrics_path,
        out_json_path=verdict_path,
    )

    metrics["oof_path"] = str(oof_path)
    metrics["analysis_oof_path"] = str(analysis_oof_path)
    metrics["candidate_metrics_path"] = str(candidate_metrics_path)
    metrics["reference_metrics_path"] = str(reference_metrics_path)
    metrics["transfer_train_path"] = str(transfer_train_path)
    metrics["transfer_test_path"] = str(transfer_test_path)
    metrics["validation_verdict_path"] = str(verdict_path)
    metrics["validation_verdict"] = verdict["verdict"]
    metrics["validation_overall_metrics"] = verdict["overall_metrics"]
    metrics.pop("oof_pred", None)
    metrics.pop("transfer_train", None)
    metrics.pop("transfer_test", None)
    write_json(metrics_path, metrics)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
