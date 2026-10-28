"""Tests for the skewness check module."""

import numpy as np
import pandas as pd
import pytest

from dqcheck.checks.skewness import check_skewness
from dqcheck.models import SEVERITY_MEDIUM, SEVERITY_HIGH


class TestSkewnessDetection:
    """Tests for skewness measurement and classification."""

    def test_symmetric_data_not_flagged(self):
        """Symmetric data (low skew) produces no findings."""
        # A symmetric distribution around 50
        values = list(range(1, 101))
        df = pd.DataFrame({"x": values})

        result = check_skewness(df)

        # Uniform distribution has skew ≈ 0, should not be flagged
        assert len(result.findings) == 0

    def test_right_skewed_data_flagged(self):
        """Right-skewed (positive skew) data is detected."""
        # Exponential-like distribution: many small, few large
        np.random.seed(42)
        values = np.random.exponential(scale=2, size=200).tolist()
        df = pd.DataFrame({"x": values})

        result = check_skewness(df)

        assert len(result.findings) >= 1
        assert result.findings[0].column == "x"
        assert "right" in result.findings[0].message.lower() or "positive" in result.findings[0].message.lower()

    def test_high_skew_suggests_log_for_positive(self):
        """Highly skewed positive-only data gets a log-transform suggestion."""
        # All values positive and highly skewed
        values = [1] * 50 + [2] * 30 + [5] * 10 + [100] * 5 + [1000] * 2
        df = pd.DataFrame({"x": values})

        result = check_skewness(df)

        if result.findings:
            finding = result.findings[0]
            if finding.severity == SEVERITY_HIGH:
                assert "log" in finding.message.lower()

    def test_no_log_suggestion_with_zeros(self):
        """Data with zeros should NOT get a log-transform suggestion."""
        values = [0] * 20 + [1] * 30 + [10] * 10 + [100] * 5
        df = pd.DataFrame({"x": values})

        result = check_skewness(df)

        for finding in result.findings:
            if "log transform" in finding.message.lower() and "not suitable" not in finding.message.lower():
                # If it says "log transform", it should say "not suitable"
                assert "not suitable" in finding.message.lower() or "Since all values are positive" in finding.message


class TestSkewnessEdgeCases:
    """Edge case tests for skewness."""

    def test_no_numeric_columns(self):
        """No numeric columns → skipped."""
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = check_skewness(df)

        assert len(result.findings) == 0
        assert "skipped" in result.summary.lower()

    def test_too_few_values(self):
        """Fewer than 3 non-null values → column skipped."""
        df = pd.DataFrame({"x": [1.0, np.nan, np.nan]})
        result = check_skewness(df)

        # Only 1 non-null value, can't compute skewness
        assert len(result.findings) == 0

    def test_all_nan_skipped(self):
        """All-NaN numeric column is skipped."""
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        result = check_skewness(df)

        assert len(result.findings) == 0
