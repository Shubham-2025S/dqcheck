"""
CLI entry point for dqcheck.

Usage:
    python -m dqcheck data.csv [options]

This module wires together the loader, all check modules, and the report
generator into a single pipeline. It uses argparse for argument parsing
and produces either Markdown (default) or HTML output.

Exit codes:
    0 — Success (report generated).
    1 — Bad input (file not found, empty file, invalid arguments).
"""

import argparse
import sys
from typing import List, Optional

import pandas as pd

from dqcheck.loader import load_csv
from dqcheck.checks.missing import check_missing
from dqcheck.checks.duplicates import check_duplicates
from dqcheck.checks.outliers import check_outliers
from dqcheck.checks.skewness import check_skewness
from dqcheck.checks.correlation import check_correlation
from dqcheck.checks.structure import check_structure
from dqcheck.report import (
    build_markdown_report,
    build_html_report,
    calculate_health_score,
)
from dqcheck.models import ReportData, CheckResult


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.

    Separated from main() so tests can inspect the parser without
    actually running the CLI.
    """
    parser = argparse.ArgumentParser(
        prog="dqcheck",
        description=(
            "dqcheck — CSV Data Quality Checker.\n"
            "Analyses a CSV file and produces a report answering: "
            '"How much can I trust this data before building a model?"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "csv_file",
        help="Path to the CSV file to analyse.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help=(
            "Comma-separated list of key columns for duplicate checking. "
            'Example: --key "id,date"'
        ),
    )
    parser.add_argument(
        "--format",
        choices=["md", "html"],
        default="md",
        dest="output_format",
        help="Output format: md (Markdown, default) or html.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to write the report to. If omitted, prints to stdout."
        ),
    )
    parser.add_argument(
        "--outlier-method",
        choices=["iqr", "zscore"],
        default="iqr",
        help="Outlier detection method: iqr (default) or zscore.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.9,
        help=(
            "Absolute correlation threshold for flagging pairs. "
            "Default: 0.9."
        ),
    )
    parser.add_argument(
        "--missing-tokens",
        type=str,
        default=None,
        help=(
            "Comma-separated extra tokens to treat as missing values. "
            'Example: --missing-tokens "?,-, unknown"'
        ),
    )

    return parser


def run_pipeline(
    csv_path: str,
    key_columns: Optional[List[str]],
    outlier_method: str,
    corr_threshold: float,
    missing_tokens: Optional[List[str]],
) -> ReportData:
    """Run the full data-quality pipeline and return the report data.

    This is the core function that coordinates all checks. It is
    separated from the CLI so it can be called programmatically
    or from tests.

    Args:
        csv_path:       Path to the CSV file.
        key_columns:    Optional list of key columns for duplicate checking.
        outlier_method: "iqr" or "zscore".
        corr_threshold: Correlation threshold for flagging.
        missing_tokens: Extra tokens to treat as hidden missing values.

    Returns:
        A fully populated ReportData object.
    """
    # Step 1: Load the CSV
    df, dataset_info = load_csv(csv_path)

    # Step 2: Run all checks in pipeline order
    check_results: List[CheckResult] = []

    check_results.append(check_missing(df, extra_tokens=missing_tokens))
    check_results.append(check_duplicates(df, key_columns=key_columns))
    check_results.append(check_outliers(df, method=outlier_method))
    check_results.append(check_skewness(df))
    check_results.append(check_correlation(df, threshold=corr_threshold))
    check_results.append(check_structure(df))

    # Step 3: Calculate health score
    score, high, medium, low, info = calculate_health_score(check_results)

    # Step 4: Assemble report data
    report = ReportData(
        dataset_info=dataset_info,
        check_results=check_results,
        health_score=score,
        high_count=high,
        medium_count=medium,
        low_count=low,
        info_count=info,
    )

    return report


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 for success, 1 for errors.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Parse key columns
    key_columns = None
    if args.key:
        key_columns = [k.strip() for k in args.key.split(",")]

    # Parse extra missing tokens
    missing_tokens = None
    if args.missing_tokens:
        missing_tokens = [t.strip() for t in args.missing_tokens.split(",")]

    # Run the pipeline
    try:
        report = run_pipeline(
            csv_path=args.csv_file,
            key_columns=key_columns,
            outlier_method=args.outlier_method,
            corr_threshold=args.corr_threshold,
            missing_tokens=missing_tokens,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Build the report
    if args.output_format == "html":
        report_text = build_html_report(report)
    else:
        report_text = build_markdown_report(report)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report written to {args.output}")
    else:
        print(report_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
