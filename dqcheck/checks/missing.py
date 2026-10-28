"""
Missing-values check for dqcheck.

Detects both *obvious* missing values (NaN/None that pandas already
recognises) and *hidden* missing values — sentinel strings like "?",
"-", "N/A", etc. that pandas treats as regular data.

Why detect hidden missing values?
  Many CSV files exported from spreadsheets or legacy systems use
  placeholder strings instead of leaving cells truly empty. If you
  feed these into a model without cleaning, they will corrupt numeric
  columns or create spurious categories in categorical columns.

Severity thresholds (per column):
  - <5%   → low   (minor, unlikely to hurt model performance)
  - 5-30% → medium (consider imputation or investigate the cause)
  - >30%  → high  (serious: the column may not be usable as-is)
  - >50%  → also add a recommendation to consider dropping the column
"""

from typing import List, Optional

import pandas as pd

from dqcheck.models import CheckResult, Finding, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH


# These sentinel values look like real data to pandas but actually mean
# "no value". We compare against lowercase, whitespace-stripped versions.
DEFAULT_HIDDEN_TOKENS = {
    "", " ", "?", "-", "na", "n/a", "null", "none", "nan",
}


def check_missing(
    df: pd.DataFrame,
    extra_tokens: Optional[List[str]] = None,
) -> CheckResult:
    """Run the missing-values check on every column.

    Args:
        df:           The DataFrame to analyse.
        extra_tokens: Additional strings to treat as hidden missing values.
                      For example, a user might pass ["unknown", "tbd"].

    Returns:
        A CheckResult containing one Finding per column that has any
        missing or hidden-missing values.
    """
    tokens = _build_token_set(extra_tokens)
    result = CheckResult(name="Missing Values")
    findings: List[Finding] = []

    for col in df.columns:
        finding = _check_column_missing(df, col, tokens)
        if finding is not None:
            findings.append(finding)

    result.findings = findings
    result.summary = _build_summary(findings, len(df.columns))
    return result


def _build_token_set(extra_tokens: Optional[List[str]]) -> set:
    """Merge the default hidden-missing tokens with any user-supplied extras.

    All tokens are stored lowercase so comparison is case-insensitive.
    """
    tokens = set(DEFAULT_HIDDEN_TOKENS)
    if extra_tokens:
        for token in extra_tokens:
            tokens.add(token.strip().lower())
    return tokens


def _check_column_missing(
    df: pd.DataFrame,
    col: str,
    tokens: set,
) -> Optional[Finding]:
    """Analyse a single column for missing and hidden-missing values.

    Steps:
      1. Count pandas NaN/None values.
      2. Among the non-NaN values, check if any match our hidden tokens.
      3. Sum the two counts to get total missing.
      4. Assign severity based on percentage thresholds.

    Returns:
        A Finding if any missing values were found, else None.
    """
    total_rows = len(df)
    if total_rows == 0:
        return None

    # Step 1: Count standard NaN/None
    pandas_missing = int(df[col].isna().sum())

    # Step 2: Count hidden missing values in non-NaN cells
    hidden_missing = _count_hidden_missing(df[col], tokens)

    # Step 3: Total
    total_missing = pandas_missing + hidden_missing
    if total_missing == 0:
        return None

    percentage = (total_missing / total_rows) * 100
    severity = _classify_severity(percentage)

    # Step 4: Build the message
    message = _build_message(
        col, pandas_missing, hidden_missing, total_missing, percentage, total_rows
    )

    metric = f"{total_missing}/{total_rows} ({percentage:.1f}%)"

    return Finding(
        column=col,
        check="missing",
        severity=severity,
        metric=metric,
        message=message,
    )


def _count_hidden_missing(series: pd.Series, tokens: set) -> int:
    """Count non-NaN values that match hidden-missing tokens.

    We cast each value to string, strip whitespace, and lowercase it
    before comparing. This catches things like " NA ", "  ?  ", etc.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0

    count = 0
    for value in non_null:
        cleaned = str(value).strip().lower()
        if cleaned in tokens:
            count += 1
    return count


def _classify_severity(percentage: float) -> str:
    """Map a missing-value percentage to a severity level.

    Thresholds:
      <5%   → low
      5-30% → medium
      >30%  → high
    """
    if percentage > 30:
        return SEVERITY_HIGH
    if percentage >= 5:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def _build_message(
    col: str,
    pandas_missing: int,
    hidden_missing: int,
    total_missing: int,
    percentage: float,
    total_rows: int,
) -> str:
    """Create a plain-English explanation of the missing-values finding.

    Includes a recommendation if more than 50% of values are missing.
    """
    parts = []
    parts.append(
        f"Column '{col}' has {total_missing} missing values "
        f"({percentage:.1f}% of {total_rows} rows)."
    )

    if pandas_missing > 0 and hidden_missing > 0:
        parts.append(
            f"  {pandas_missing} are standard NaN/None, "
            f"and {hidden_missing} are hidden missing values "
            f"(sentinel strings like '?', 'N/A', etc.)."
        )
    elif hidden_missing > 0:
        parts.append(
            f"  All {hidden_missing} are hidden missing values "
            f"(sentinel strings like '?', 'N/A', etc.)."
        )

    if percentage > 50:
        parts.append(
            "  Consider dropping this column — over half the values are missing, "
            "which may make imputation unreliable."
        )

    return " ".join(parts)


def _build_summary(findings: List[Finding], total_columns: int) -> str:
    """Create a one-line summary for the report header."""
    if not findings:
        return f"No missing values found across {total_columns} columns."

    high = sum(1 for f in findings if f.severity == SEVERITY_HIGH)
    med = sum(1 for f in findings if f.severity == SEVERITY_MEDIUM)
    low = sum(1 for f in findings if f.severity == SEVERITY_LOW)
    return (
        f"Found missing values in {len(findings)}/{total_columns} columns "
        f"({high} high, {med} medium, {low} low severity)."
    )
