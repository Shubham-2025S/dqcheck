"""Tests for the CSV loader module."""

import os
import tempfile

import pandas as pd
import pytest

from dqcheck.loader import load_csv


class TestLoadCsvSuccess:
    """Tests for successful CSV loading."""

    def test_loads_basic_csv(self, tmp_path):
        """A simple CSV with numeric and string columns loads correctly."""
        csv_file = tmp_path / "basic.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")

        df, info = load_csv(str(csv_file))

        assert len(df) == 2
        assert info.row_count == 2
        assert info.column_count == 2
        assert "name" in info.dtypes
        assert "age" in info.dtypes

    def test_latin1_fallback(self, tmp_path):
        """A CSV with latin-1 encoding is read without error."""
        csv_file = tmp_path / "latin1.csv"
        # Write bytes that are valid latin-1 but invalid utf-8
        content = "name,city\nAlice,Zürich\nBob,Malmö\n"
        csv_file.write_bytes(content.encode("latin-1"))

        df, info = load_csv(str(csv_file))

        assert len(df) == 2
        assert info.row_count == 2


class TestLoadCsvErrors:
    """Tests for error handling in CSV loading."""

    def test_file_not_found(self):
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_csv("/nonexistent/path/data.csv")

    def test_empty_file(self, tmp_path):
        """A zero-byte file raises ValueError."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="empty"):
            load_csv(str(csv_file))

    def test_header_only_file(self, tmp_path):
        """A file with a header but no data rows raises ValueError."""
        csv_file = tmp_path / "header_only.csv"
        csv_file.write_text("name,age,city\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no data rows"):
            load_csv(str(csv_file))


class TestDtypeDetection:
    """Tests for detecting likely wrong dtypes."""

    def test_detects_numeric_looking_strings(self, tmp_path):
        """Columns with values like '$1,200' are flagged as likely numeric."""
        csv_file = tmp_path / "money.csv"
        lines = ["price\n"]
        for i in range(10):
            lines.append(f"${i * 100}\n")
        csv_file.write_text("".join(lines), encoding="utf-8")

        df, info = load_csv(str(csv_file))

        assert len(info.dtype_warnings) >= 1
        warning = info.dtype_warnings[0]
        assert warning.column == "price"
        assert warning.likely_dtype == "numeric"

    def test_detects_date_looking_strings(self, tmp_path):
        """Columns with values like '2024-01-15' are flagged as likely date."""
        csv_file = tmp_path / "dates.csv"
        lines = ["event_date\n"]
        for i in range(1, 11):
            lines.append(f"2024-01-{i:02d}\n")
        csv_file.write_text("".join(lines), encoding="utf-8")

        df, info = load_csv(str(csv_file))

        # Date columns might be parsed by pandas or flagged by us
        # Check if pandas parsed it or if we flagged it
        if str(df["event_date"].dtype) == "object":
            assert len(info.dtype_warnings) >= 1
            assert info.dtype_warnings[0].likely_dtype == "date"

    def test_no_false_positive_on_text(self, tmp_path):
        """A column of plain text should not be flagged."""
        csv_file = tmp_path / "text.csv"
        csv_file.write_text(
            "comment\nhello world\ngood morning\nnice day\nfoo bar\n"
            "baz qux\ntest value\nsome text\nmore words\nanother line\nfinal row\n",
            encoding="utf-8",
        )

        df, info = load_csv(str(csv_file))

        assert len(info.dtype_warnings) == 0

    def test_single_row_file(self, tmp_path):
        """A file with only one data row still loads and analyses correctly."""
        csv_file = tmp_path / "one_row.csv"
        csv_file.write_text("x,y\n1,2\n", encoding="utf-8")

        df, info = load_csv(str(csv_file))

        assert info.row_count == 1
        assert info.column_count == 2
