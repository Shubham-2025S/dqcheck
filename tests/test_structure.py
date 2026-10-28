"""Tests for the structure check module."""

import pandas as pd
import pytest

from dqcheck.checks.structure import check_structure
from dqcheck.models import SEVERITY_HIGH, SEVERITY_MEDIUM


class TestConstantColumns:
    """Tests for constant-column detection."""

    def test_constant_column_flagged(self):
        """A column with one unique value is flagged as constant."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "const": ["X", "X", "X"],
        })

        result = check_structure(df)

        const_findings = [
            f for f in result.findings if "constant" in f.message.lower()
        ]
        assert len(const_findings) == 1
        assert const_findings[0].column == "const"
        assert const_findings[0].severity == SEVERITY_HIGH

    def test_all_nan_column_flagged(self):
        """A column that is entirely NaN has nunique=0 and is flagged."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "empty": [None, None, None],
        })

        result = check_structure(df)

        empty_findings = [f for f in result.findings if f.column == "empty"]
        assert len(empty_findings) == 1
        assert empty_findings[0].severity == SEVERITY_HIGH


class TestIDLikeColumns:
    """Tests for ID-like column detection."""

    def test_id_column_flagged(self):
        """A column named 'id' with unique values is flagged as ID-like."""
        df = pd.DataFrame({
            "id": list(range(100)),
            "value": list(range(100, 200)),
        })

        result = check_structure(df)

        id_findings = [
            f for f in result.findings if "identifier" in f.message.lower()
        ]
        assert len(id_findings) == 1
        assert id_findings[0].column == "id"
        assert id_findings[0].severity == SEVERITY_MEDIUM

    def test_customer_id_flagged(self):
        """A column named 'customer_id' with unique values is flagged."""
        df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(100)],
            "name": ["Alice"] * 100,
        })

        result = check_structure(df)

        id_findings = [
            f for f in result.findings
            if "identifier" in f.message.lower() and f.column == "customer_id"
        ]
        assert len(id_findings) == 1

    def test_non_id_column_not_flagged(self):
        """A column with unique values but a non-ID name is not flagged."""
        df = pd.DataFrame({
            "temperature": list(range(100)),
            "pressure": list(range(100, 200)),
        })

        result = check_structure(df)

        id_findings = [
            f for f in result.findings if "identifier" in f.message.lower()
        ]
        assert len(id_findings) == 0

    def test_normal_column_not_flagged(self):
        """A non-constant, non-unique column produces no structure findings."""
        df = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B"] * 20,
        })

        result = check_structure(df)

        assert len(result.findings) == 0
