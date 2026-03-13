#!/usr/bin/env python3
"""Join Kaggle submission history with local submission artifacts."""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from churn_baseline.submission_forensics import write_submission_forensics_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join Kaggle submission history with local submission artifacts")
    parser.add_argument("--competition", default="playground-series-s6e3")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument(
        "--submissions-dir",
        default="artifacts/submissions",
        help="Local directory scanned for submission CSV existence. Report linking uses the CSV basename referenced inside report JSONs.",
    )
    parser.add_argument("--out-summary-json", default="artifacts/reports/submission_forensics_summary.json")
    parser.add_argument("--out-ledger-csv", default="artifacts/reports/submission_forensics_ledger.csv")
    parser.add_argument("--out-reports-csv", default="artifacts/reports/submission_forensics_report_links.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_submission_forensics_outputs(
        competition=args.competition,
        reports_dir=args.reports_dir,
        submissions_dir=args.submissions_dir,
        out_summary_json=args.out_summary_json,
        out_ledger_csv=args.out_ledger_csv,
        out_reports_csv=args.out_reports_csv,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
