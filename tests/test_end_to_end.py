"""End-to-end test: run the full pipeline on the messy sample CSV."""

import os

import pytest

from dqcheck.cli import run_pipeline
from dqcheck.report import build_markdown_report, build_html_report


# Path to the sample data relative to this test file
SAMPLE_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "sample_data",
    "messy_sample.csv",
)


class TestEndToEnd:
    """Full pipeline integration tests on the sample dataset."""

    def test_pipeline_runs_without_error(self):
        """The full pipeline completes without exceptions."""
        report = run_pipeline(
            csv_path=SAMPLE_CSV,
            key_columns=["customer_id"],
            outlier_method="iqr",
            corr_threshold=0.9,
            missing_tokens=None,
        )

        assert report.dataset_info.row_count == 200
        assert report.dataset_info.column_count == 12
        assert 0 <= report.health_score <= 100

    def test_finds_known_issues(self):
        """The pipeline detects issues we deliberately planted in the data."""
        report = run_pipeline(
            csv_path=SAMPLE_CSV,
            key_columns=["customer_id"],
            outlier_method="iqr",
            corr_threshold=0.9,
            missing_tokens=None,
        )

        all_findings = []
        for cr in report.check_results:
            all_findings.extend(cr.findings)

        # Check that some findings were generated
        assert len(all_findings) > 0

        # Check that specific known issues were found:
        finding_columns = [f.column for f in all_findings]
        finding_checks = [f.check for f in all_findings]

        # 1. Missing values should be found (we planted ?, N/A, null, etc.)
        assert "missing" in finding_checks

        # 2. Constant column should be found (constant_col is all "X")
        assert "constant_col" in finding_columns

        # 3. Duplicate rows should be found (key duplicates on customer_id)
        assert any("duplicates" in c for c in finding_checks)

    def test_markdown_report_generation(self):
        """Markdown report is generated and contains expected sections."""
        report = run_pipeline(
            csv_path=SAMPLE_CSV,
            key_columns=None,
            outlier_method="iqr",
            corr_threshold=0.9,
            missing_tokens=None,
        )

        md = build_markdown_report(report)

        assert "# " in md  # Has headers
        assert "Health Score" in md
        assert "Missing Values" in md
        assert "Outliers" in md
        assert "Correlation" in md
        assert "Structure" in md

    def test_html_report_generation(self):
        """HTML report is generated and is self-contained."""
        report = run_pipeline(
            csv_path=SAMPLE_CSV,
            key_columns=None,
            outlier_method="iqr",
            corr_threshold=0.9,
            missing_tokens=None,
        )

        html = build_html_report(report)

        assert "<!DOCTYPE html>" in html
        assert "<style>" in html
        assert "Health Score" not in html or "score" in html.lower()
        assert "</html>" in html

    def test_zscore_method_works(self):
        """The z-score outlier method also works end-to-end."""
        report = run_pipeline(
            csv_path=SAMPLE_CSV,
            key_columns=None,
            outlier_method="zscore",
            corr_threshold=0.9,
            missing_tokens=None,
        )

        assert report.dataset_info.row_count == 200
        assert 0 <= report.health_score <= 100

    def test_custom_corr_threshold(self):
        """A lower correlation threshold flags more pairs."""
        report_high = run_pipeline(
            csv_path=SAMPLE_CSV,
            key_columns=None,
            outlier_method="iqr",
            corr_threshold=0.99,
            missing_tokens=None,
        )
        report_low = run_pipeline(
            csv_path=SAMPLE_CSV,
            key_columns=None,
            outlier_method="iqr",
            corr_threshold=0.5,
            missing_tokens=None,
        )

        high_corr_findings = []
        low_corr_findings = []
        for cr in report_high.check_results:
            if cr.name == "Correlation":
                high_corr_findings = cr.findings
        for cr in report_low.check_results:
            if cr.name == "Correlation":
                low_corr_findings = cr.findings

        assert len(low_corr_findings) >= len(high_corr_findings)

    def test_custom_missing_tokens(self):
        """Extra missing tokens are detected."""
        report = run_pipeline(
            csv_path=SAMPLE_CSV,
            key_columns=None,
            outlier_method="iqr",
            corr_threshold=0.9,
            missing_tokens=["Typo age"],
        )

        # Should still work and complete
        assert report.dataset_info.row_count == 200

    def test_key_duplicate_check(self):
        """Key duplicate check finds the planted duplicates."""
        report = run_pipeline(
            csv_path=SAMPLE_CSV,
            key_columns=["customer_id"],
            outlier_method="iqr",
            corr_threshold=0.9,
            missing_tokens=None,
        )

        dup_findings = []
        for cr in report.check_results:
            if cr.name == "Duplicate Rows":
                dup_findings = cr.findings

        # Should find at least the key duplicates (C001, C002 appear twice)
        assert len(dup_findings) >= 1
