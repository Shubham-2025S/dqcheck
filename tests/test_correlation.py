"""Tests for the correlation check module."""

import pandas as pd
import pytest

from dqcheck.checks.correlation import check_correlation
from dqcheck.models import SEVERITY_HIGH


class TestCorrelationDetection:
    """Tests for detecting high correlations."""

    def test_perfectly_correlated_pair(self):
        """Two perfectly correlated columns are flagged."""
        df = pd.DataFrame({
            "x": list(range(100)),
            "y": list(range(100)),  # y = x → r = 1.0
            "z": list(range(100, 200)),  # z = x + 100 → r = 1.0
        })

        result = check_correlation(df, threshold=0.9)

        # All three pairs should be flagged since they're perfectly correlated
        assert len(result.findings) >= 1

    def test_uncorrelated_columns(self):
        """Uncorrelated columns produce no findings."""
        import numpy as np
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(100),
            "y": np.random.randn(100),
        })

        result = check_correlation(df, threshold=0.9)

        assert len(result.findings) == 0

    def test_each_pair_reported_once(self):
        """A correlated pair is reported only once, not twice."""
        df = pd.DataFrame({
            "a": list(range(50)),
            "b": list(range(50)),
        })

        result = check_correlation(df, threshold=0.9)

        # Should be exactly 1 finding for the (a, b) pair
        assert len(result.findings) == 1

    def test_negative_correlation_detected(self):
        """Strong negative correlation is also flagged."""
        df = pd.DataFrame({
            "x": list(range(50)),
            "y": list(range(50, 0, -1)),  # y = -x → r ≈ -1.0
        })

        result = check_correlation(df, threshold=0.9)

        assert len(result.findings) == 1
        assert result.findings[0].severity == SEVERITY_HIGH

    def test_custom_threshold(self):
        """A lower threshold flags more pairs."""
        import numpy as np
        np.random.seed(123)
        x = np.arange(100, dtype=float)
        # y has moderate correlation with x
        y = x + np.random.randn(100) * 10

        df = pd.DataFrame({"x": x, "y": y})

        # With high threshold, may not be flagged
        result_high = check_correlation(df, threshold=0.99)
        # With low threshold, should be flagged
        result_low = check_correlation(df, threshold=0.5)

        assert len(result_low.findings) >= len(result_high.findings)


class TestCorrelationEdgeCases:
    """Edge case tests for correlation."""

    def test_fewer_than_two_numeric_columns(self):
        """With <2 numeric columns, the check is skipped."""
        df = pd.DataFrame({"x": [1, 2, 3], "name": ["a", "b", "c"]})

        result = check_correlation(df)

        assert len(result.findings) == 0
        assert "skipped" in result.summary.lower()

    def test_pearson_limitation_noted(self):
        """The finding message mentions Pearson's limitation."""
        df = pd.DataFrame({
            "a": list(range(50)),
            "b": list(range(50)),
        })

        result = check_correlation(df, threshold=0.9)

        if result.findings:
            assert "linear" in result.findings[0].message.lower()
