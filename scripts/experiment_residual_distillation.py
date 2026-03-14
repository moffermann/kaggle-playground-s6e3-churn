"""CLI for residual-family distillation train/predict flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.config import CatBoostHyperParams
from churn_baseline.feature_engineering import normalize_feature_blocks
from churn_baseline.residual_distillation import (
    DISTILLATION_BASE_REFERENCE_SUBMISSION_PATH,
    ResidualDistillationConfig,
    make_residual_distillation_submission,
    run_total_residual_distillation_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train or materialize the total residual distillation line: "
            "learn v3 - base_reference and optionally apply it over a base teacher submission."
        )
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "submit"],
        help="Train the distillation regressor or materialize a submission from an existing model.",
    )
    parser.add_argument("--train-csv", default="data/raw/train.csv", help="Competition train CSV.")
    parser.add_argument("--test-csv", default="data/raw/test.csv", help="Competition test CSV for mode=submit.")
    parser.add_argument("--out-dir", default="artifacts/reports", help="Directory for analysis and metrics.")
    parser.add_argument("--model-dir", default="artifacts/models", help="Directory for the CatBoostRegressor artifact.")
    parser.add_argument("--label", default="residual_distillation_smoke", help="Artifact prefix label.")
    parser.add_argument(
        "--feature-blocks",
        default="H,R,S,V",
        help="Comma-separated feature blocks for the distilled regressor.",
    )
    parser.add_argument(
        "--alpha-grid",
        default="0.25,0.5,1.0,2.0,4.0",
        help="Comma-separated alpha scan over base_reference + alpha * distilled_delta_pred.",
    )
    parser.add_argument(
        "--reference-submission",
        default="artifacts/submissions/playground-series-s6e3-rvblend.csv",
        help="Base teacher submission used as reference in mode=submit.",
    )
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional metrics JSON to source best_alpha/feature_blocks/model_path for mode=submit.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional explicit model path for mode=submit. Defaults to the path stored in metrics-json or <model-dir>/<label>.cbm.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional explicit alpha override for mode=submit.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional output submission CSV path for mode=submit.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional report JSON path for mode=submit.",
    )
    parser.add_argument("--folds", type=int, default=2, help="Number of CV folds for the smoke.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--iterations", type=int, default=150, help="CatBoostRegressor iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="CatBoostRegressor learning rate.")
    parser.add_argument("--depth", type=int, default=6, help="CatBoostRegressor depth.")
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0, help="CatBoostRegressor L2 regularization.")
    parser.add_argument("--early-stopping-rounds", type=int, default=60, help="Early stopping rounds.")
    parser.add_argument("--verbose", type=int, default=0, help="CatBoost verbosity.")
    return parser


def _parse_alpha_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in str(raw).split(",") if token.strip())
    if not values:
        raise ValueError("alpha-grid must contain at least one numeric value")
    return values


def _load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _default_metrics_path(label: str, out_dir: str) -> Path:
    return Path(out_dir) / f"{label}_metrics.json"


def _default_output_csv(label: str) -> Path:
    return Path("artifacts/submissions") / f"playground-series-s6e3-{label}.csv"


def _default_report_json(label: str, out_dir: str) -> Path:
    return Path(out_dir) / f"submission_candidate_{label}.json"


def _resolve_submit_train_csv(
    *,
    cli_train_csv: str,
    metrics_payload: dict[str, object],
) -> Path:
    metrics_train_csv = str(metrics_payload.get("train_csv_path", "")).strip()
    if not metrics_train_csv:
        return Path(cli_train_csv)
    cli_path = Path(cli_train_csv).resolve()
    metrics_path = Path(metrics_train_csv).resolve()
    if cli_path != metrics_path:
        raise ValueError(
            "mode=submit requires the same train_csv_path used during training: "
            f"expected '{metrics_path}', got '{cli_path}'."
        )
    return metrics_path


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "train":
        feature_blocks = tuple(
            normalize_feature_blocks([token.strip() for token in str(args.feature_blocks).split(",") if token.strip()])
        )
        config = ResidualDistillationConfig(
            label=str(args.label),
            feature_blocks=feature_blocks,
            alpha_grid=_parse_alpha_grid(args.alpha_grid),
        )
        params = CatBoostHyperParams(
            iterations=int(args.iterations),
            learning_rate=float(args.learning_rate),
            depth=int(args.depth),
            l2_leaf_reg=float(args.l2_leaf_reg),
            random_seed=int(args.random_state),
        )
        summary = run_total_residual_distillation_smoke(
            train_csv_path=args.train_csv,
            out_dir=args.out_dir,
            model_dir=args.model_dir,
            config=config,
            params=params,
            folds=int(args.folds),
            random_state=int(args.random_state),
            early_stopping_rounds=int(args.early_stopping_rounds),
            verbose=int(args.verbose),
        )
        print(json.dumps(summary, indent=2))
        return

    metrics_path = (
        Path(args.metrics_json) if str(args.metrics_json).strip() else _default_metrics_path(str(args.label), args.out_dir)
    )
    metrics_payload = _load_json(metrics_path)
    metrics_feature_blocks = metrics_payload.get("feature_blocks")
    if not isinstance(metrics_feature_blocks, list):
        raise ValueError(f"{metrics_path} must contain 'feature_blocks' for mode=submit")
    feature_blocks = tuple(normalize_feature_blocks(metrics_feature_blocks))
    alpha = float(args.alpha if args.alpha is not None else metrics_payload["best_alpha"])
    model_path = Path(str(args.model_path).strip()) if str(args.model_path).strip() else Path(str(metrics_payload["model_path"]))
    train_csv_path = _resolve_submit_train_csv(cli_train_csv=args.train_csv, metrics_payload=metrics_payload)
    expected_reference_submission_path = Path(
        str(metrics_payload.get("expected_reference_submission_path", DISTILLATION_BASE_REFERENCE_SUBMISSION_PATH))
    ).resolve()
    supplied_reference_submission_path = Path(str(args.reference_submission).strip()).resolve()
    if supplied_reference_submission_path != expected_reference_submission_path:
        raise ValueError(
            "mode=submit requires the same reference submission lineage used during training: "
            f"expected '{expected_reference_submission_path}', got '{supplied_reference_submission_path}'."
        )
    output_csv = Path(str(args.output_csv).strip()) if str(args.output_csv).strip() else _default_output_csv(str(args.label))
    report_json = Path(str(args.report_json).strip()) if str(args.report_json).strip() else _default_report_json(str(args.label), args.out_dir)

    report = make_residual_distillation_submission(
        test_csv_path=args.test_csv,
        train_csv_path=train_csv_path,
        reference_submission_path=supplied_reference_submission_path,
        expected_reference_submission_path=expected_reference_submission_path,
        model_path=model_path,
        feature_blocks=feature_blocks,
        alpha=alpha,
        output_csv_path=output_csv,
        report_json_path=report_json,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
