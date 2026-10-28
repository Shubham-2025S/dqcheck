"""Tests for the duplicates check module."""

import pandas as pd
import pytest

from dqcheck.checks.duplicates import check_duplicates
from dqcheck.models import SEVERITY_LOW, SEVERITY_HIGH, SEVERITY_INFO


class TestFullRowDuplicates:
    """Tests for full-row duplicate detection."""

    def test_no_duplicates(self):
        """No findings when every row is unique."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = check_duplicates(df)

        dup_findings = [f for f in result.findings if f.check == "duplicates"]
        assert len(dup_findings) == 0

    def test_exact_duplicates_found(self):
        """Exact duplicate rows are detected and counted."""
        df = pd.DataFrame({
            "a": [1, 2, 1, 3],
            "b": ["x", "y", "x", "z"],
        })
        result = check_duplicates(df)

        dup_findings = [f for f in result.findings if f.check == "duplicates"]
        assert len(dup_findings) == 1
        assert "1/4" in dup_findings[0].metric

    def test_single_row_no_duplicates(self):
        """A DataFrame with only one row has no duplicates."""
        df = pd.DataFrame({"a": [1]})
        result = check_duplicates(df)

        assert len(result.findings) == 0


class TestKeyDuplicates:
    """Tests for key-based duplicate detection."""

    def test_key_duplicates_with_conflicts(self):
        """Same key, different values → high severity."""
        df = pd.DataFrame({
            "id": [1, 2, 1, 3],
            "value": ["a", "b", "DIFFERENT", "c"],
        })
        result = check_duplicates(df, key_columns=["id"])

        key_findings = [f for f in result.findings if "key" in f.check]
        assert len(key_findings) == 1
        assert key_findings[0].severity == SEVERITY_HIGH

    def test_key_duplicates_without_conflicts(self):
        """Same key, same values → low severity (just full dups on that key)."""
        df = pd.DataFrame({
            "id": [1, 2, 1, 3],
            "value": ["a", "b", "a", "c"],
        })
        result = check_duplicates(df, key_columns=["id"])

        key_findings = [f for f in result.findings if "key" in f.check]
        assert len(key_findings) == 1
        assert key_findings[0].severity == SEVERITY_LOW

    def test_missing_key_column(self):
        """A key column that doesn't exist produces an info finding."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = check_duplicates(df, key_columns=["nonexistent"])

        key_findings = [f for f in result.findings if "key" in f.check]
        assert len(key_findings) == 1
        assert key_findings[0].severity == SEVERITY_INFO

    def test_no_key_no_key_check(self):
        """When no key columns are provided, no key check runs."""
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "z"]})
        result = check_duplicates(df, key_columns=None)

        key_findings = [f for f in result.findings if "key" in f.check]
        assert len(key_findings) == 0
