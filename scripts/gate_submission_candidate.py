#!/usr/bin/env python3
"""Gate a submission candidate using diagnostics thresholds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.artifacts import write_json


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _check(
    name: str,
    status: str,
    message: str,
    *,
    actual: Any = None,
    threshold: Any = None,
) -> dict[str, Any]:
    if status not in {PASS, WARN, FAIL}:
        raise ValueError(f"Invalid status: {status}")
    return {
        "name": name,
        "status": status,
        "message": message,
        "actual": actual,
        "threshold": threshold,
    }


def _status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for row in checks:
        status = str(row.get("status", "")).upper()
        if status in counts:
            counts[status] += 1
    return {
        "pass": int(counts[PASS]),
        "warn": int(counts[WARN]),
        "fail": int(counts[FAIL]),
        "total": int(sum(counts.values())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate a submission candidate from diagnostics reports")
    parser.add_argument(
        "--candidate-name",
        default="playground-series-s6e3",
        help="Candidate identifier for report traceability.",
    )
    parser.add_argument(
        "--parity-json",
        default="artifacts/reports/diagnostic_submission_parity_issue5.json",
        help="Path to parity report JSON.",
    )
    parser.add_argument(
        "--drift-json",
        default="artifacts/reports/diagnostic_train_test_drift_issue5.json",
        help="Path to drift report JSON.",
    )
    parser.add_argument(
        "--robustness-json",
        default="artifacts/reports/diagnostic_ensemble_robustness_issue5.json",
        help="Path to robustness report JSON.",
    )
    parser.add_argument(
        "--require-parity-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require parity overall_status=PASS.",
    )
    parser.add_argument(
        "--max-adv-auc",
        type=float,
        default=0.58,
        help="Max allowed adversarial validation AUC.",
    )
    parser.add_argument(
        "--max-high-psi-count",
        type=int,
        default=2,
        help="Max allowed count of numeric columns with PSI >= 0.2.",
    )
    parser.add_argument(
        "--max-high-tvd-count",
        type=int,
        default=2,
        help="Max allowed count of categorical columns with TVD >= 0.2.",
    )
    parser.add_argument(
        "--min-weighted-delta-vs-baseline",
        type=float,
        default=0.0002,
        help="Minimum required weighted CV mean improvement vs baseline.",
    )
    parser.add_argument(
        "--min-weighted-delta-vs-equal",
        type=float,
        default=0.00005,
        help="Minimum required weighted CV mean improvement vs equal blend.",
    )
    parser.add_argument(
        "--max-weighted-cv-std",
        type=float,
        default=0.0015,
        help="Max allowed weighted CV std.",
    )
    parser.add_argument(
        "--max-weighted-optimism",
        type=float,
        default=0.00005,
        help="Max allowed weighted optimism (full_auc - cv_mean_auc).",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=0,
        help="Maximum warnings tolerated for GO decision.",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/reports/diagnostic_submission_gate_issue5.json",
        help="Output gating report JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parity = _load_json(args.parity_json)
    drift = _load_json(args.drift_json)
    robustness = _load_json(args.robustness_json)

    checks: list[dict[str, Any]] = []

    parity_status = str(parity.get("overall_status", "")).upper()
    if args.require_parity_pass:
        status = PASS if parity_status == PASS else FAIL
        checks.append(
            _check(
                "parity_overall_status",
                status,
                "Parity report must be PASS.",
                actual=parity_status,
                threshold=PASS,
            )
        )
    else:
        checks.append(
            _check(
                "parity_overall_status",
                WARN if parity_status != PASS else PASS,
                "Parity PASS is recommended.",
                actual=parity_status,
                threshold=PASS,
            )
        )

    adv_auc = float(drift["adversarial_validation"]["cv_mean_auc"])
    checks.append(
        _check(
            "drift_adversarial_auc",
            PASS if adv_auc <= args.max_adv_auc else FAIL,
            "Adversarial AUC should remain below threshold.",
            actual=adv_auc,
            threshold=float(args.max_adv_auc),
        )
    )

    high_psi_count = int(drift["numeric_drift"]["high_psi_count_ge_0_2"])
    checks.append(
        _check(
            "drift_high_psi_count",
            PASS if high_psi_count <= args.max_high_psi_count else FAIL,
            "Count of high-PSI numeric features should remain controlled.",
            actual=high_psi_count,
            threshold=int(args.max_high_psi_count),
        )
    )

    high_tvd_count = int(drift["categorical_drift"]["high_tvd_count_ge_0_2"])
    checks.append(
        _check(
            "drift_high_tvd_count",
            PASS if high_tvd_count <= args.max_high_tvd_count else FAIL,
            "Count of high-TVD categorical features should remain controlled.",
            actual=high_tvd_count,
            threshold=int(args.max_high_tvd_count),
        )
    )

    methods = robustness.get("methods", {})
    if "weighted" not in methods or "equal" not in methods:
        checks.append(
            _check(
                "robustness_methods_presence",
                FAIL,
                "Robustness report must include 'weighted' and 'equal' methods.",
                actual=list(methods.keys()),
                threshold=["weighted", "equal"],
            )
        )
    else:
        weighted = methods["weighted"]
        equal = methods["equal"]
        rank = methods.get("rank")

        weighted_cv_mean = float(weighted["cv_mean_auc"])
        weighted_cv_std = float(weighted["cv_std_auc"])
        weighted_optimism = float(weighted["optimism_full_minus_cv_mean"])

        delta_vs_baseline = weighted.get("delta_cv_mean_vs_baseline_auc")
        if delta_vs_baseline is None:
            checks.append(
                _check(
                    "weighted_delta_vs_baseline_present",
                    FAIL,
                    "Weighted delta vs baseline is required; rerun robustness with baseline metrics.",
                    actual=None,
                    threshold="float",
                )
            )
        else:
            delta_vs_baseline = float(delta_vs_baseline)
            checks.append(
                _check(
                    "weighted_delta_vs_baseline",
                    PASS if delta_vs_baseline >= args.min_weighted_delta_vs_baseline else FAIL,
                    "Weighted CV mean improvement vs baseline must exceed threshold.",
                    actual=delta_vs_baseline,
                    threshold=float(args.min_weighted_delta_vs_baseline),
                )
            )

        delta_vs_equal = float(weighted_cv_mean - float(equal["cv_mean_auc"]))
        checks.append(
            _check(
                "weighted_delta_vs_equal",
                PASS if delta_vs_equal >= args.min_weighted_delta_vs_equal else WARN,
                "Weighted CV mean should improve over equal blend.",
                actual=delta_vs_equal,
                threshold=float(args.min_weighted_delta_vs_equal),
            )
        )

        if rank is not None:
            delta_vs_rank = float(weighted_cv_mean - float(rank["cv_mean_auc"]))
            checks.append(
                _check(
                    "weighted_delta_vs_rank",
                    PASS if delta_vs_rank >= 0.0 else WARN,
                    "Weighted CV mean should not underperform rank blend.",
                    actual=delta_vs_rank,
                    threshold=">= 0.0",
                )
            )

        checks.append(
            _check(
                "weighted_cv_std",
                PASS if weighted_cv_std <= args.max_weighted_cv_std else WARN,
                "Weighted CV std should remain below threshold.",
                actual=weighted_cv_std,
                threshold=float(args.max_weighted_cv_std),
            )
        )

        checks.append(
            _check(
                "weighted_optimism",
                PASS if weighted_optimism <= args.max_weighted_optimism else WARN,
                "Weighted optimism should remain controlled.",
                actual=weighted_optimism,
                threshold=float(args.max_weighted_optimism),
            )
        )

    counts = _status_counts(checks)
    decision = "GO" if counts["fail"] == 0 and counts["warn"] <= args.max_warnings else "NO_GO"

    report = {
        "generated_at_utc": _now_utc_iso(),
        "candidate_name": args.candidate_name,
        "decision": decision,
        "inputs": {
            "parity_json": args.parity_json,
            "drift_json": args.drift_json,
            "robustness_json": args.robustness_json,
        },
        "thresholds": {
            "require_parity_pass": bool(args.require_parity_pass),
            "max_adv_auc": float(args.max_adv_auc),
            "max_high_psi_count": int(args.max_high_psi_count),
            "max_high_tvd_count": int(args.max_high_tvd_count),
            "min_weighted_delta_vs_baseline": float(args.min_weighted_delta_vs_baseline),
            "min_weighted_delta_vs_equal": float(args.min_weighted_delta_vs_equal),
            "max_weighted_cv_std": float(args.max_weighted_cv_std),
            "max_weighted_optimism": float(args.max_weighted_optimism),
            "max_warnings": int(args.max_warnings),
        },
        "summary": counts,
        "checks": checks,
    }
    write_json(args.out_json, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
