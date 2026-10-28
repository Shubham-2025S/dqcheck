"""
Structure check for dqcheck.

Flags two kinds of structurally useless columns:
  1. Constant columns: columns with 0 or 1 unique values. These carry
     zero information and will not help any model learn patterns.
  2. ID-like columns: columns where almost every row has a unique value
     AND the column name suggests it is an identifier (contains "id",
     "index", "key", "code", "uuid", "pk", etc.). These are useful for
     lookups but should not be used as features — a model cannot
     generalise from unique identifiers.

Why this matters:
  Leaving constant or ID columns in a training dataset wastes compute,
  can confuse feature-importance measures, and (for IDs) risks the model
  memorising row identifiers instead of learning real patterns.
"""

import re
from typing import List

import pandas as pd

from dqcheck.models import CheckResult, Finding, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_INFO


# Patterns that suggest a column is an identifier.
# We check the column name (lowercased) against these.
_ID_NAME_PATTERNS = [
    "id", "uuid", "guid", "index", "idx", "key", "pk",
    "code", "number", "num", "no", "serial",
]


def check_structure(df: pd.DataFrame) -> CheckResult:
    """Run structure checks: constant and ID-like columns.

    Args:
        df: The DataFrame to analyse.

    Returns:
        A CheckResult with findings for structurally useless columns.
    """
    result = CheckResult(name="Structure (Constant & ID-like Columns)")
    findings: List[Finding] = []

    total_rows = len(df)

    for col in df.columns:
        nunique = df[col].nunique(dropna=True)

        # Check 1: Constant columns (0 or 1 unique values)
        constant_finding = _check_constant(col, nunique)
        if constant_finding is not None:
            findings.append(constant_finding)
            continue  # No need to also check if it is ID-like

        # Check 2: ID-like columns
        id_finding = _check_id_like(col, nunique, total_rows, df[col].dtype)
        if id_finding is not None:
            findings.append(id_finding)

    result.findings = findings
    result.summary = _build_summary(findings)
    return result


def _check_constant(col: str, nunique: int) -> "Finding | None":
    """Flag a column as constant if it has 0 or 1 unique values.

    A column with 0 unique values means it is entirely NaN/null.
    A column with 1 unique value means every row has the same value.
    Both carry zero information.
    """
    if nunique > 1:
        return None

    if nunique == 0:
        message = (
            f"Column '{col}' is entirely empty (all values are NaN/null). "
            f"It carries zero information and should be dropped."
        )
    else:
        message = (
            f"Column '{col}' has only one unique value — it is constant. "
            f"A constant column carries zero information and will not help "
            f"any model learn patterns. Consider dropping it."
        )

    return Finding(
        column=col,
        check="structure",
        severity=SEVERITY_HIGH,
        metric=f"nunique={nunique}",
        message=message,
    )


def _check_id_like(
    col: str,
    nunique: int,
    total_rows: int,
    dtype: "pd.api.types.pandas_dtype",
) -> "Finding | None":
    """Flag a column as ID-like if it has near-unique values and an ID-like name.

    Criteria (both must be true):
      1. The number of unique values is >= 90% of the row count.
      2. The column name matches an ID-like pattern OR the dtype is integer/object
         (strings and integers are the most common ID types).

    We require both conditions to avoid false positives on columns like
    "price" that happen to have many unique values.
    """
    if total_rows == 0:
        return None

    uniqueness_ratio = nunique / total_rows
    if uniqueness_ratio < 0.9:
        return None

    # Check if the column name suggests an identifier
    name_is_id_like = _name_looks_like_id(col)

    if not name_is_id_like:
        return None

    percentage = uniqueness_ratio * 100

    message = (
        f"Column '{col}' looks like an identifier "
        f"({nunique}/{total_rows} unique values = {percentage:.1f}%). "
        f"ID columns should not be used as features — a model cannot "
        f"generalise from unique identifiers. Use it for joins or lookups only."
    )

    return Finding(
        column=col,
        check="structure",
        severity=SEVERITY_MEDIUM,
        metric=f"uniqueness={percentage:.1f}%",
        message=message,
    )


def _name_looks_like_id(col_name: str) -> bool:
    """Check if a column name matches common identifier naming patterns.

    We use simple substring/word-boundary matching, not full NLP. This
    catches the vast majority of ID columns in real datasets.
    """
    lower_name = col_name.lower().strip()

    # Exact match for very short names like "id"
    if lower_name in ("id", "pk", "idx"):
        return True

    # Check if any ID pattern appears as a word boundary in the name.
    # For example, "customer_id" matches, but "grid" does not match "id"
    # because we require word boundaries.
    for pattern in _ID_NAME_PATTERNS:
        # Use word boundary regex to avoid false positives
        regex = r"(?:^|[_\-\s])" + re.escape(pattern) + r"(?:$|[_\-\s])"
        if re.search(regex, lower_name):
            return True

    # Also check if the name ends with "_id" or "Id" (common convention)
    if lower_name.endswith("_id") or lower_name.endswith("id"):
        # But avoid matching words like "grid", "valid", "bid"
        # by requiring at least a separator before "id" or name == "id"
        prefix = lower_name[: -2]
        if prefix == "" or prefix.endswith(("_", "-", " ")):
            return True

    return True if lower_name.endswith("_key") or lower_name.endswith("_code") else False


def _build_summary(findings: List[Finding]) -> str:
    """One-line summary for the report."""
    if not findings:
        return "No constant or ID-like columns found."

    constants = sum(1 for f in findings if "constant" in f.message.lower() or "empty" in f.message.lower())
    ids = sum(1 for f in findings if "identifier" in f.message.lower())

    parts = []
    if constants:
        parts.append(f"{constants} constant column(s)")
    if ids:
        parts.append(f"{ids} ID-like column(s)")
    return f"Found {' and '.join(parts)}."
