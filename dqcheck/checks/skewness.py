"""
Skewness check for dqcheck.

Measures the skewness (asymmetry) of each numeric column's distribution.

Why skewness matters:
  Many statistical models and algorithms (e.g. linear regression, k-means)
  assume that features are roughly symmetrically distributed. Highly skewed
  data can lead to poor model performance because the mean is pulled away
  from the centre of the distribution, giving extreme values outsized
  influence on the model.

Thresholds:
  |skew| > 1   → high   (strongly skewed; transformation recommended)
  0.5 ≤ |skew| ≤ 1 → medium (moderately skewed; worth investigating)
  |skew| < 0.5 → not flagged (approximately symmetric)

We also report the mean-vs-median gap as a simpler way to see skewness:
  if the mean and median are far apart relative to the standard deviation,
  the distribution is lopsided.
"""

from typing import List

import numpy as np
import pandas as pd

from dqcheck.models import CheckResult, Finding, SEVERITY_MEDIUM, SEVERITY_HIGH


def check_skewness(df: pd.DataFrame) -> CheckResult:
    """Compute skewness for every numeric column and flag the skewed ones.

    Args:
        df: The DataFrame to analyse.

    Returns:
        A CheckResult with findings for columns whose |skew| >= 0.5.
    """
    result = CheckResult(name="Skewness")
    findings: List[Finding] = []

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        result.summary = "No numeric columns found — skewness check skipped."
        return result

    for col in numeric_columns:
        finding = _check_column_skewness(df[col], col)
        if finding is not None:
            findings.append(finding)

    result.findings = findings
    result.summary = _build_summary(findings, len(numeric_columns))
    return result


def _check_column_skewness(
    series: pd.Series, col_name: str
) -> "Finding | None":
    """Compute skewness for a single numeric column.

    Uses pandas .skew() which computes the unbiased Fisher–Pearson
    coefficient of skewness. A positive value means a right tail;
    a negative value means a left tail.

    We skip columns with fewer than 3 non-null values because skewness
    is not meaningful with so few data points.
    """
    clean = series.dropna()

    # Need at least 3 values for skewness to be meaningful
    if len(clean) < 3:
        return None

    skew_value = float(clean.skew())
    abs_skew = abs(skew_value)

    # Only flag if |skew| >= 0.5
    if abs_skew < 0.5:
        return None

    severity = _classify_skewness_severity(abs_skew)
    mean_val = float(clean.mean())
    median_val = float(clean.median())
    mean_median_gap = mean_val - median_val

    message = _build_message(
        col_name, skew_value, mean_val, median_val, mean_median_gap, clean
    )

    metric = f"skew={skew_value:.2f}"

    return Finding(
        column=col_name,
        check="skewness",
        severity=severity,
        metric=metric,
        message=message,
    )


def _classify_skewness_severity(abs_skew: float) -> str:
    """Classify skewness severity.

    |skew| > 1  → high (strongly skewed)
    0.5 ≤ |skew| ≤ 1 → medium (moderately skewed)
    """
    if abs_skew > 1:
        return SEVERITY_HIGH
    return SEVERITY_MEDIUM


def _build_message(
    col_name: str,
    skew_value: float,
    mean_val: float,
    median_val: float,
    mean_median_gap: float,
    clean: pd.Series,
) -> str:
    """Create a plain-English explanation of the skewness finding.

    Includes:
    - The skew direction and magnitude.
    - The mean vs median gap.
    - A log-transform suggestion if all values are positive.
    """
    direction = "right (positive)" if skew_value > 0 else "left (negative)"

    parts = []
    parts.append(
        f"Column '{col_name}' is {direction}-skewed (skew={skew_value:.2f}). "
        f"Mean={mean_val:.4g}, Median={median_val:.4g} "
        f"(gap={mean_median_gap:.4g})."
    )

    # Suggest log transform only if every value is strictly positive,
    # because log(0) and log(negative) are undefined.
    all_positive = bool((clean > 0).all())
    if all_positive and abs(skew_value) > 1:
        parts.append(
            "Since all values are positive, a log transform may help "
            "reduce skewness before modelling."
        )
    elif abs(skew_value) > 1:
        parts.append(
            "A log transform is not suitable here because the column "
            "contains zero or negative values. Consider other transforms "
            "(e.g. square root for non-negative data, or Yeo–Johnson)."
        )

    return " ".join(parts)


def _build_summary(findings: List[Finding], total_numeric: int) -> str:
    """One-line summary for the report."""
    if not findings:
        return f"All {total_numeric} numeric columns have low skewness (|skew| < 0.5)."

    high = sum(1 for f in findings if f.severity == SEVERITY_HIGH)
    med = sum(1 for f in findings if f.severity == SEVERITY_MEDIUM)
    return (
        f"Found skewed distributions in {len(findings)}/{total_numeric} "
        f"numeric columns ({high} high, {med} medium)."
    )
