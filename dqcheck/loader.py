"""
CSV loader for dqcheck.

Reads a CSV file into a pandas DataFrame, handling encoding issues,
edge cases (empty files, header-only files, missing paths), and
inspecting column dtypes to flag likely misclassifications.

Design decisions:
- Try utf-8 first because it is the most common encoding. Fall back to
  latin-1 which can read any byte sequence, so we never crash on encoding.
- We intentionally never modify the original DataFrame — no dropping,
  no casting. The loader only *reports* what it finds.
- We flag "numeric-looking" and "date-looking" object columns by sampling
  values and applying simple heuristic regex patterns. This avoids heavy
  parsing while still catching the most common cases ("$1,200", "12%",
  "2024-01-15").
"""

import os
import re
from typing import List, Tuple

import pandas as pd

from dqcheck.models import DatasetInfo, DtypeWarning


def load_csv(file_path: str) -> Tuple[pd.DataFrame, DatasetInfo]:
    """Load a CSV file and return the DataFrame plus dataset metadata.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A tuple of (DataFrame, DatasetInfo).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is empty (zero bytes).
        ValueError: If the file has a header but no data rows.
    """
    _validate_path(file_path)
    df = _read_with_fallback_encoding(file_path)
    _validate_dataframe(df, file_path)

    dtype_warnings = _detect_dtype_issues(df)

    dtypes_dict = {col: str(df[col].dtype) for col in df.columns}

    info = DatasetInfo(
        file_path=file_path,
        row_count=len(df),
        column_count=len(df.columns),
        dtypes=dtypes_dict,
        dtype_warnings=dtype_warnings,
    )

    return df, info


# ---------------------------------------------------------------------------
# Private helpers — each does exactly one thing.
# ---------------------------------------------------------------------------


def _validate_path(file_path: str) -> None:
    """Check that the file exists and is not empty (zero bytes).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file has zero bytes.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty (zero bytes): {file_path}")


def _read_with_fallback_encoding(file_path: str) -> pd.DataFrame:
    """Try reading the CSV as utf-8; if that fails, fall back to latin-1.

    latin-1 can decode any byte sequence, so this two-step approach
    handles virtually every CSV a user might throw at the tool.
    """
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin-1")


def _validate_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """Ensure the loaded DataFrame has at least one data row.

    A header-only CSV is valid to *read* but useless to *analyse*, so we
    reject it early with a clear message.

    Raises:
        ValueError: If the DataFrame has zero rows.
    """
    if len(df) == 0:
        raise ValueError(
            f"File has a header but no data rows: {file_path}"
        )


def _detect_dtype_issues(df: pd.DataFrame) -> List[DtypeWarning]:
    """Scan object columns for values that look numeric or date-like.

    We only inspect columns whose pandas dtype is 'object' (i.e. stored as
    strings). For each one we sample up to 50 non-null values and check them
    against simple regex patterns.

    Returns:
        A list of DtypeWarning objects for any suspicious columns.
    """
    warnings: List[DtypeWarning] = []

    object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_columns:
        non_null_values = df[col].dropna().astype(str)
        if len(non_null_values) == 0:
            continue

        sample = non_null_values.head(50)
        warning = _check_column_dtype(col, sample)
        if warning is not None:
            warnings.append(warning)

    return warnings


# Patterns for detecting numeric-looking strings.
# Matches things like: "1,200", "$50.00", "12%", "-3.5", "+100"
_NUMERIC_PATTERN = re.compile(
    r"^\s*[+\-]?\s*[\$€£¥]?\s*[\d,]+\.?\d*\s*%?\s*$"
)

# Patterns for detecting date-looking strings.
# Matches common date formats: "2024-01-15", "01/15/2024", "15-Jan-2024"
_DATE_PATTERN = re.compile(
    r"^\s*\d{1,4}[\-/\.]\d{1,2}[\-/\.]\d{1,4}\s*$"
)


def _check_column_dtype(
    col_name: str, sample: pd.Series
) -> "DtypeWarning | None":
    """Check a sample of string values to see if they look numeric or date-like.

    We require at least 60% of the sample to match the pattern before
    flagging the column. This avoids false positives on mixed-content columns.

    Args:
        col_name: The name of the column.
        sample:   A pandas Series of string values (up to 50).

    Returns:
        A DtypeWarning if the column is likely miscategorised, else None.
    """
    total = len(sample)
    if total == 0:
        return None

    # Check numeric first (more common issue than date)
    numeric_matches = sample.apply(lambda v: bool(_NUMERIC_PATTERN.match(str(v))))
    numeric_count = numeric_matches.sum()
    if numeric_count / total >= 0.6:
        examples = sample[numeric_matches].head(3).tolist()
        return DtypeWarning(
            column=col_name,
            current_dtype="object",
            likely_dtype="numeric",
            evidence=f"Examples: {examples}",
        )

    # Check date
    date_matches = sample.apply(lambda v: bool(_DATE_PATTERN.match(str(v))))
    date_count = date_matches.sum()
    if date_count / total >= 0.6:
        examples = sample[date_matches].head(3).tolist()
        return DtypeWarning(
            column=col_name,
            current_dtype="object",
            likely_dtype="date",
            evidence=f"Examples: {examples}",
        )

    return None
