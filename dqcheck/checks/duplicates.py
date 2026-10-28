"""
Duplicate-rows check for dqcheck.

Detects two kinds of duplicates:
  1. Full-row duplicates: rows where every column value is identical.
  2. Key duplicates: rows that share the same value(s) in user-specified
     key columns but differ in at least one other column. This is important
     because it suggests data-entry errors or a join gone wrong.

Why this matters:
  Duplicate rows inflate your dataset and can bias a model by giving
  certain observations more weight. Key duplicates are even trickier —
  if two rows share a primary key but have different values, at least
  one of them is wrong.
"""

from typing import List, Optional

import pandas as pd

from dqcheck.models import CheckResult, Finding, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_INFO


def check_duplicates(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
) -> CheckResult:
    """Run the duplicates check.

    Args:
        df:          The DataFrame to analyse.
        key_columns: If provided, also check for rows with the same key
                     values but different other values.

    Returns:
        A CheckResult with findings about duplicate rows.
    """
    result = CheckResult(name="Duplicate Rows")
    findings: List[Finding] = []

    # Full-row duplicates
    full_dup_finding = _check_full_row_duplicates(df)
    if full_dup_finding is not None:
        findings.append(full_dup_finding)

    # Key duplicates (only if user provided key columns)
    if key_columns:
        key_finding = _check_key_duplicates(df, key_columns)
        if key_finding is not None:
            findings.append(key_finding)

    result.findings = findings
    result.summary = _build_summary(findings)
    return result


def _check_full_row_duplicates(df: pd.DataFrame) -> Optional[Finding]:
    """Count rows that are exact duplicates of another row.

    We use pandas .duplicated() which marks the *second and subsequent*
    occurrences as duplicates (keeps the first). This is the standard
    convention.
    """
    total_rows = len(df)
    if total_rows <= 1:
        return None

    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return None

    percentage = (dup_count / total_rows) * 100
    severity = _classify_full_dup_severity(percentage)

    message = (
        f"Found {dup_count} exact duplicate rows ({percentage:.1f}% of {total_rows} rows). "
        f"Duplicate rows can bias models by over-representing certain observations."
    )

    return Finding(
        column="(all columns)",
        check="duplicates",
        severity=severity,
        metric=f"{dup_count}/{total_rows} ({percentage:.1f}%)",
        message=message,
    )


def _check_key_duplicates(
    df: pd.DataFrame, key_columns: List[str]
) -> Optional[Finding]:
    """Find rows that share key values but differ in other columns.

    Steps:
      1. Check that all key columns exist in the DataFrame.
      2. Find groups where the same key appears more than once.
      3. Among those groups, check if values in non-key columns differ.

    A key duplicate with different values is always high severity because
    it implies a data integrity problem.
    """
    # Validate that all key columns exist
    missing_keys = [k for k in key_columns if k not in df.columns]
    if missing_keys:
        return Finding(
            column=", ".join(key_columns),
            check="duplicates (key)",
            severity=SEVERITY_INFO,
            metric="N/A",
            message=f"Key columns not found in data: {missing_keys}. Skipping key duplicate check.",
        )

    total_rows = len(df)
    if total_rows <= 1:
        return None

    # Find rows where the key columns have duplicate values
    key_dup_mask = df.duplicated(subset=key_columns, keep=False)
    key_dup_count = int(key_dup_mask.sum())

    if key_dup_count == 0:
        return None

    # Check if any of those key-duplicate groups have different non-key values
    non_key_columns = [c for c in df.columns if c not in key_columns]
    conflicting_count = _count_conflicting_key_groups(
        df, key_columns, non_key_columns, key_dup_mask
    )

    percentage = (key_dup_count / total_rows) * 100
    key_str = ", ".join(key_columns)

    if conflicting_count > 0:
        severity = SEVERITY_HIGH
        message = (
            f"Found {key_dup_count} rows ({percentage:.1f}%) sharing the same "
            f"key ({key_str}), and {conflicting_count} key groups have "
            f"conflicting values in other columns. This suggests data "
            f"integrity issues — the same entity has different recorded values."
        )
    else:
        severity = SEVERITY_LOW
        message = (
            f"Found {key_dup_count} rows ({percentage:.1f}%) sharing the same "
            f"key ({key_str}), but all non-key values are identical. "
            f"These are full duplicates with respect to those keys."
        )

    return Finding(
        column=key_str,
        check="duplicates (key)",
        severity=severity,
        metric=f"{key_dup_count}/{total_rows} ({percentage:.1f}%)",
        message=message,
    )


def _count_conflicting_key_groups(
    df: pd.DataFrame,
    key_columns: List[str],
    non_key_columns: List[str],
    key_dup_mask: pd.Series,
) -> int:
    """Count how many key groups have differing values in non-key columns.

    We group the duplicate rows by the key columns, then check if there
    is more than one unique value in any non-key column within each group.
    """
    if not non_key_columns:
        return 0

    dup_rows = df[key_dup_mask]
    grouped = dup_rows.groupby(key_columns, dropna=False)

    conflicting = 0
    for _key, group in grouped:
        for col in non_key_columns:
            unique_values = group[col].dropna().nunique()
            if unique_values > 1:
                conflicting += 1
                break  # One conflicting column is enough to flag the group

    return conflicting


def _classify_full_dup_severity(percentage: float) -> str:
    """Classify full-row duplicate severity by percentage.

    <1%  → low  (a few duplicates, common in real data)
    1-5% → medium
    >5%  → high (significant portion of data is duplicated)
    """
    if percentage > 5:
        return SEVERITY_HIGH
    if percentage >= 1:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def _build_summary(findings: List[Finding]) -> str:
    """Create a one-line summary for the report."""
    if not findings:
        return "No duplicate rows found."
    parts = [f.metric for f in findings]
    return f"Duplicate issues found: {'; '.join(parts)}."
