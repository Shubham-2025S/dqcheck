"""Tests for the outlier detection module."""

import numpy as np
import pandas as pd
import pytest

from dqcheck.checks.outliers import check_outliers
from dqcheck.models import SEVERITY_INFO


class TestIQRMethod:
    """Tests for IQR-based outlier detection."""

    def test_detects_outliers(self):
        """Obvious outliers are detected with the IQR method."""
        # Normal-ish data with one extreme outlier
        values = list(range(1, 21)) + [1000]
        df = pd.DataFrame({"x": values})

        result = check_outliers(df, method="iqr")

        outlier_findings = [
            f for f in result.findings if f.severity != SEVERITY_INFO
        ]
        assert len(outlier_findings) >= 1
        assert outlier_findings[0].column == "x"

    def test_no_outliers_in_tight_data(self):
        """Data with no extreme values produces no outlier findings."""
        df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 15]})
        result = check_outliers(df, method="iqr")

        outlier_findings = [
            f for f in result.findings if f.severity != SEVERITY_INFO
        ]
        assert len(outlier_findings) == 0

    def test_iqr_zero_skipped(self):
        """When IQR is zero (constant data), the column is skipped with info."""
        df = pd.DataFrame({"x": [5, 5, 5, 5, 5]})
        result = check_outliers(df, method="iqr")

        info_findings = [
            f for f in result.findings if f.severity == SEVERITY_INFO
        ]
        assert len(info_findings) == 1
        assert "IQR=0" in info_findings[0].metric


class TestZScoreMethod:
    """Tests for z-score-based outlier detection."""

    def test_detects_outliers(self):
        """Obvious outliers are detected with the z-score method."""
        values = list(range(1, 51)) + [500]
        df = pd.DataFrame({"x": values})

        result = check_outliers(df, method="zscore")

        outlier_findings = [
            f for f in result.findings if f.severity != SEVERITY_INFO
        ]
        assert len(outlier_findings) >= 1

    def test_std_zero_skipped(self):
        """When std is zero (constant data), the column is skipped with info."""
        df = pd.DataFrame({"x": [7, 7, 7, 7, 7]})
        result = check_outliers(df, method="zscore")

        info_findings = [
            f for f in result.findings if f.severity == SEVERITY_INFO
        ]
        assert len(info_findings) == 1
        assert "std=0" in info_findings[0].metric


class TestEdgeCases:
    """Edge case tests for outlier detection."""

    def test_all_nan_column_skipped(self):
        """An all-NaN column produces no findings."""
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        result = check_outliers(df, method="iqr")

        assert len(result.findings) == 0

    def test_no_numeric_columns(self):
        """A DataFrame with only string columns skips the outlier check."""
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
        result = check_outliers(df, method="iqr")

        assert len(result.findings) == 0
        assert "skipped" in result.summary.lower()

    def test_message_says_not_necessarily_error(self):
        """The outlier message includes the 'not necessarily errors' caveat."""
        values = list(range(1, 21)) + [1000]
        df = pd.DataFrame({"x": values})

        result = check_outliers(df, method="iqr")

        outlier_findings = [
            f for f in result.findings if f.severity != SEVERITY_INFO
        ]
        if outlier_findings:
            assert "not necessarily" in outlier_findings[0].message.lower()
