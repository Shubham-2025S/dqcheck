"""Tests for the report generation module."""

import pandas as pd
import pytest

from dqcheck.models import (
    DatasetInfo,
    CheckResult,
    Finding,
    ReportData,
    DtypeWarning,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO,
)
from dqcheck.report import (
    calculate_health_score,
    build_markdown_report,
    build_html_report,
)


def _make_report() -> ReportData:
    """Build a small ReportData for testing."""
    info = DatasetInfo(
        file_path="test.csv",
        row_count=100,
        column_count=5,
        dtypes={"a": "int64", "b": "float64", "c": "object", "d": "int64", "e": "object"},
        dtype_warnings=[
            DtypeWarning(
                column="e", current_dtype="object",
                likely_dtype="numeric", evidence="Examples: ['$100']",
            ),
        ],
    )

    missing_result = CheckResult(
        name="Missing Values",
        findings=[
            Finding(column="c", check="missing", severity=SEVERITY_HIGH,
                    metric="40/100 (40.0%)", message="Column 'c' has 40 missing values."),
            Finding(column="b", check="missing", severity=SEVERITY_LOW,
                    metric="2/100 (2.0%)", message="Column 'b' has 2 missing values."),
        ],
        summary="Found missing values in 2/5 columns.",
    )

    outlier_result = CheckResult(
        name="Outliers",
        findings=[
            Finding(column="a", check="outliers", severity=SEVERITY_MEDIUM,
                    metric="3/100 (3.0%)", message="Column 'a' has 3 outliers."),
        ],
        summary="Found outliers in 1/4 numeric columns.",
    )

    empty_result = CheckResult(
        name="Duplicates",
        findings=[],
        summary="No duplicate rows found.",
    )

    check_results = [missing_result, outlier_result, empty_result]
    score, high, med, low, info_count = calculate_health_score(check_results)

    return ReportData(
        dataset_info=info,
        check_results=check_results,
        health_score=score,
        high_count=high,
        medium_count=med,
        low_count=low,
        info_count=info_count,
    )


class TestHealthScore:
    """Tests for the health score calculation."""

    def test_perfect_score(self):
        """No findings → score of 100."""
        results = [CheckResult(name="Test", findings=[])]
        score, h, m, l, i = calculate_health_score(results)
        assert score == 100

    def test_deductions(self):
        """High, medium, and low findings deduct correctly."""
        findings = [
            Finding(column="a", check="test", severity=SEVERITY_HIGH,
                    metric="", message=""),
            Finding(column="b", check="test", severity=SEVERITY_MEDIUM,
                    metric="", message=""),
            Finding(column="c", check="test", severity=SEVERITY_LOW,
                    metric="", message=""),
        ]
        results = [CheckResult(name="Test", findings=findings)]
        score, h, m, l, i = calculate_health_score(results)

        # 100 - 10 - 3 - 1 = 86
        assert score == 86
        assert h == 1
        assert m == 1
        assert l == 1

    def test_minimum_zero(self):
        """Score cannot go below 0."""
        findings = [
            Finding(column=f"col{i}", check="test", severity=SEVERITY_HIGH,
                    metric="", message="")
            for i in range(20)
        ]
        results = [CheckResult(name="Test", findings=findings)]
        score, _, _, _, _ = calculate_health_score(results)

        assert score == 0

    def test_info_no_deduction(self):
        """Info-level findings do not reduce the score."""
        findings = [
            Finding(column="a", check="test", severity=SEVERITY_INFO,
                    metric="", message=""),
        ]
        results = [CheckResult(name="Test", findings=findings)]
        score, _, _, _, i = calculate_health_score(results)

        assert score == 100
        assert i == 1


class TestMarkdownReport:
    """Tests for Markdown report generation."""

    def test_contains_header(self):
        """The report contains the dqcheck header."""
        report = _make_report()
        md = build_markdown_report(report)
        assert "dqcheck" in md

    def test_contains_health_score(self):
        """The report contains the health score."""
        report = _make_report()
        md = build_markdown_report(report)
        assert f"{report.health_score}/100" in md

    def test_contains_check_sections(self):
        """The report contains a section for each check."""
        report = _make_report()
        md = build_markdown_report(report)
        assert "Missing Values" in md
        assert "Outliers" in md
        assert "Duplicates" in md

    def test_contains_dtype_warnings(self):
        """The report includes dtype warnings."""
        report = _make_report()
        md = build_markdown_report(report)
        assert "Dtype Warnings" in md

    def test_contains_dataset_info(self):
        """The report contains row and column counts."""
        report = _make_report()
        md = build_markdown_report(report)
        assert "100" in md  # row count
        assert "5" in md    # column count


class TestHTMLReport:
    """Tests for HTML report generation."""

    def test_valid_html_structure(self):
        """The HTML starts with DOCTYPE and has head/body tags."""
        report = _make_report()
        html = build_html_report(report)
        assert html.startswith("<!DOCTYPE html>")
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html

    def test_contains_inline_css(self):
        """The HTML has a <style> block (inline CSS)."""
        report = _make_report()
        html = build_html_report(report)
        assert "<style>" in html

    def test_contains_severity_badges(self):
        """The HTML contains colour-coded severity badges."""
        report = _make_report()
        html = build_html_report(report)
        assert "badge-high" in html
        assert "badge-medium" in html

    def test_no_external_dependencies(self):
        """The HTML does not reference external stylesheets or scripts."""
        report = _make_report()
        html = build_html_report(report)
        assert '<link rel="stylesheet"' not in html
        assert '<script src=' not in html

    def test_contains_health_score(self):
        """The HTML contains the health score."""
        report = _make_report()
        html = build_html_report(report)
        assert f"{report.health_score}/100" in html
