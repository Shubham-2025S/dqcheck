"""Tests for the missing-values check module."""

import pandas as pd
import pytest

from dqcheck.checks.missing import check_missing
from dqcheck.models import SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH


class TestBasicMissing:
    """Tests for standard NaN detection."""

    def test_no_missing(self):
        """A DataFrame with no missing values produces no findings."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = check_missing(df)
        assert len(result.findings) == 0

    def test_standard_nan_detected(self):
        """Standard NaN values are counted correctly."""
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})
        result = check_missing(df)

        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.column == "a"
        assert finding.check == "missing"
        assert "2/5" in finding.metric

    def test_all_nan_column(self):
        """A column that is entirely NaN is detected as high severity."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        result = check_missing(df)

        findings_for_b = [f for f in result.findings if f.column == "b"]
        assert len(findings_for_b) == 1
        assert findings_for_b[0].severity == SEVERITY_HIGH


class TestHiddenMissing:
    """Tests for hidden missing value detection."""

    def test_question_mark(self):
        """The '?' sentinel is detected as a hidden missing value."""
        df = pd.DataFrame({"a": ["hello", "?", "world"]})
        result = check_missing(df)

        assert len(result.findings) == 1
        assert "hidden" in result.findings[0].message.lower()

    def test_na_variants(self):
        """Various NA-like strings are all detected."""
        df = pd.DataFrame({
            "a": ["val", "N/A", "n/a", "NA", "na", "null", "None", "nan", "NaN", "real"]
        })
        result = check_missing(df)

        assert len(result.findings) == 1
        finding = result.findings[0]
        # 8 hidden missing out of 10
        assert "8/10" in finding.metric

    def test_whitespace_only(self):
        """A cell with only whitespace is detected as hidden missing."""
        df = pd.DataFrame({"a": ["hello", "   ", "world"]})
        result = check_missing(df)

        assert len(result.findings) == 1

    def test_empty_string(self):
        """Empty strings are detected as hidden missing."""
        df = pd.DataFrame({"a": ["hello", "", "world"]})
        result = check_missing(df)

        assert len(result.findings) == 1

    def test_case_insensitive(self):
        """Hidden missing detection is case-insensitive."""
        df = pd.DataFrame({"a": ["ok", "NULL", "Ok", "NONE", "ok"]})
        result = check_missing(df)

        assert len(result.findings) == 1
        # "NULL" and "NONE" are hidden missing
        assert "2/5" in result.findings[0].metric


class TestMissingSeverity:
    """Tests for severity classification."""

    def test_low_severity(self):
        """<5% missing → low severity."""
        # 1 missing out of 25 = 4%
        values = list(range(24)) + [None]
        df = pd.DataFrame({"a": values})
        result = check_missing(df)

        assert result.findings[0].severity == SEVERITY_LOW

    def test_medium_severity(self):
        """5-30% missing → medium severity."""
        # 5 missing out of 20 = 25%
        values = list(range(15)) + [None] * 5
        df = pd.DataFrame({"a": values})
        result = check_missing(df)

        assert result.findings[0].severity == SEVERITY_MEDIUM

    def test_high_severity(self):
        """>30% missing → high severity."""
        # 8 missing out of 10 = 80%
        values = [1, 2] + [None] * 8
        df = pd.DataFrame({"a": values})
        result = check_missing(df)

        assert result.findings[0].severity == SEVERITY_HIGH

    def test_over_50_recommends_dropping(self):
        """>50% missing includes a 'consider dropping' recommendation."""
        values = [1, 2] + [None] * 8
        df = pd.DataFrame({"a": values})
        result = check_missing(df)

        assert "dropping" in result.findings[0].message.lower()


class TestExtraTokens:
    """Tests for user-supplied extra missing tokens."""

    def test_custom_token(self):
        """User-supplied tokens like 'unknown' are detected."""
        df = pd.DataFrame({"a": ["hello", "unknown", "world"]})
        result = check_missing(df, extra_tokens=["unknown"])

        assert len(result.findings) == 1
