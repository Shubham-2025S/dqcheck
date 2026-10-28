"""
Outlier detection for dqcheck.

Supports two methods:
  1. IQR (Interquartile Range) — the default. An outlier is any value
     below Q1 - 1.5*IQR or above Q3 + 1.5*IQR. This method is robust
     to skewed distributions and does not assume normality, which is why
     it is the default.
  2. Z-score — flags values with |z| > 3 (more than 3 standard deviations
     from the mean). This assumes the data is roughly normally distributed,
     which is often not the case for real-world data.

IMPORTANT:
  This module only *flags* outliers. It never removes or modifies them.
  An outlier is not necessarily an error — it could be a legitimate
  extreme value (e.g. a very large purchase). The report explicitly
  states this to prevent users from blindly deleting outliers.

Edge cases handled:
  - IQR == 0 (e.g. a column where Q1 == Q3): skipped with a note.
  - std == 0 (constant column): skipped with a note.
  - All-NaN columns: skipped.
  - Non-numeric columns: ignored entirely.
"""

from typing import List

import numpy as np
import pandas as pd

from dqcheck.models import CheckResult, Finding, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_INFO


def check_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
) -> CheckResult:
    """Run outlier detection on all numeric columns.

    Args:
        df:     The DataFrame to analyse.
        method: Either "iqr" (default) or "zscore".

    Returns:
        A CheckResult with one Finding per numeric column that has outliers.
    """
    result = CheckResult(name="Outliers")
    findings: List[Finding] = []

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        result.summary = "No numeric columns found — outlier check skipped."
        return result

    for col in numeric_columns:
        finding = _check_column_outliers(df[col], col, method)
        if finding is not None:
            findings.append(finding)

    result.findings = findings
    result.summary = _build_summary(findings, len(numeric_columns), method)
    return result


def _check_column_outliers(
    series: pd.Series,
    col_name: str,
    method: str,
) -> "Finding | None":
    """Detect outliers in a single numeric column.

    Dispatches to the IQR or z-score method based on the `method` argument.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return None

    if method == "zscore":
        return _zscore_check(clean, col_name)
    else:
        return _iqr_check(clean, col_name)


def _iqr_check(
    clean: pd.Series,
    col_name: str,
) -> "Finding | None":
    """Detect outliers using the IQR method.

    The IQR (Interquartile Range) is the range between the 25th and 75th
    percentiles. Values more than 1.5 × IQR below Q1 or above Q3 are
    flagged as outliers. This is the same logic used in box-and-whisker plots.

    If IQR is zero (e.g. a column where most values are the same), the
    method cannot define meaningful bounds, so we skip with an info note.
    """
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    if iqr == 0:
        return Finding(
            column=col_name,
            check="outliers",
            severity=SEVERITY_INFO,
            metric="IQR=0",
            message=(
                f"Column '{col_name}': IQR is zero (Q1 = Q3 = {q1}), so the "
                f"IQR method cannot define outlier bounds. This usually means "
                f"most values are identical. Consider checking if this column "
                f"is useful as a feature."
            ),
        )

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (clean < lower_bound) | (clean > upper_bound)
    outlier_count = int(outlier_mask.sum())

    if outlier_count == 0:
        return None

    return _build_outlier_finding(
        col_name=col_name,
        outlier_count=outlier_count,
        total=len(clean),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        actual_min=float(clean.min()),
        actual_max=float(clean.max()),
        method="IQR",
    )


def _zscore_check(
    clean: pd.Series,
    col_name: str,
) -> "Finding | None":
    """Detect outliers using the z-score method.

    A z-score measures how many standard deviations a value is from the
    mean. Values with |z| > 3 are flagged. This assumes a roughly normal
    distribution, which is often not the case, hence IQR is preferred.

    If standard deviation is zero (constant column), we skip with a note.
    """
    mean = float(clean.mean())
    std = float(clean.std())

    if std == 0:
        return Finding(
            column=col_name,
            check="outliers",
            severity=SEVERITY_INFO,
            metric="std=0",
            message=(
                f"Column '{col_name}': standard deviation is zero (all values "
                f"are {mean}). Z-score method cannot detect outliers. This is "
                f"a constant column — see the Structure check for details."
            ),
        )

    # Calculate z-scores for each value
    z_scores = (clean - mean) / std
    outlier_mask = z_scores.abs() > 3
    outlier_count = int(outlier_mask.sum())

    if outlier_count == 0:
        return None

    # Translate z-score bounds back to original scale for readability
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    return _build_outlier_finding(
        col_name=col_name,
        outlier_count=outlier_count,
        total=len(clean),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        actual_min=float(clean.min()),
        actual_max=float(clean.max()),
        method="z-score",
    )


def _build_outlier_finding(
    col_name: str,
    outlier_count: int,
    total: int,
    lower_bound: float,
    upper_bound: float,
    actual_min: float,
    actual_max: float,
    method: str,
) -> Finding:
    """Create a Finding for an outlier detection result.

    The message always includes the caveat that outliers are not
    necessarily errors.
    """
    percentage = (outlier_count / total) * 100
    severity = _classify_outlier_severity(percentage)

    message = (
        f"Column '{col_name}': {outlier_count} values ({percentage:.1f}%) "
        f"are outside the {method} bounds [{lower_bound:.2f}, {upper_bound:.2f}]. "
        f"Actual range: [{actual_min}, {actual_max}]. "
        f"Note: outliers are not necessarily errors — they may be legitimate "
        f"extreme values. Investigate before deciding whether to transform or keep them."
    )

    metric = f"{outlier_count}/{total} ({percentage:.1f}%)"

    return Finding(
        column=col_name,
        check="outliers",
        severity=severity,
        metric=metric,
        message=message,
    )


def _classify_outlier_severity(percentage: float) -> str:
    """Classify outlier severity by percentage.

    <1%  → low   (typical for clean data)
    1-5% → medium
    >5%  → high  (unusual amount of extreme values)
    """
    if percentage > 5:
        return SEVERITY_HIGH
    if percentage >= 1:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def _build_summary(
    findings: List[Finding], total_numeric: int, method: str
) -> str:
    """Create a one-line summary for the report."""
    # Separate real outlier findings from info notes
    real_findings = [f for f in findings if f.severity != SEVERITY_INFO]
    info_findings = [f for f in findings if f.severity == SEVERITY_INFO]

    if not real_findings and not info_findings:
        return f"No outliers found across {total_numeric} numeric columns (method: {method})."

    parts = []
    if real_findings:
        parts.append(
            f"Found outliers in {len(real_findings)}/{total_numeric} numeric columns"
        )
    if info_findings:
        parts.append(f"{len(info_findings)} columns skipped (zero spread)")
    return f"{'; '.join(parts)} (method: {method})."
