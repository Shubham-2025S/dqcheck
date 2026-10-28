"""
Correlation check for dqcheck.

Computes a Pearson correlation matrix on numeric columns and flags
column pairs with very high correlation (|r| > threshold) as possible
multicollinearity or redundancy.

Why this matters:
  When two features are highly correlated, they carry essentially the
  same information. Including both in a model can cause:
  - Unstable coefficients in linear models (multicollinearity).
  - Wasted computation with no extra predictive power.
  - Misleading feature-importance rankings.

Limitations (stated in every report):
  Pearson correlation only captures *linear* relationships. Two columns
  can have a perfect non-linear relationship (e.g. y = x²) and still
  show r ≈ 0. For non-linear relationships, consider mutual information
  or other measures.
"""

from typing import List

import numpy as np
import pandas as pd

from dqcheck.models import CheckResult, Finding, SEVERITY_HIGH, SEVERITY_INFO


def check_correlation(
    df: pd.DataFrame,
    threshold: float = 0.9,
) -> CheckResult:
    """Find pairs of numeric columns with high Pearson correlation.

    Args:
        df:        The DataFrame to analyse.
        threshold: Absolute correlation above which to flag a pair.
                   Default is 0.9.

    Returns:
        A CheckResult with one Finding per highly correlated pair.
    """
    result = CheckResult(name="Correlation")
    findings: List[Finding] = []

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) < 2:
        result.summary = (
            "Fewer than 2 numeric columns — correlation check skipped."
        )
        return result

    corr_matrix = df[numeric_columns].corr(method="pearson")
    flagged_pairs = _find_high_correlation_pairs(corr_matrix, threshold)

    for col_a, col_b, corr_value in flagged_pairs:
        finding = _build_correlation_finding(col_a, col_b, corr_value, threshold)
        findings.append(finding)

    result.findings = findings
    result.summary = _build_summary(findings, len(numeric_columns), threshold)
    return result


def _find_high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float,
) -> List[tuple]:
    """Extract column pairs with |correlation| above the threshold.

    We only look at the upper triangle of the matrix to avoid reporting
    each pair twice (A-B and B-A are the same relationship). We also
    skip the diagonal (a column is always perfectly correlated with itself).

    Returns:
        A list of (col_a, col_b, correlation_value) tuples.
    """
    pairs = []
    columns = corr_matrix.columns.tolist()

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr_value = corr_matrix.iloc[i, j]
            # Skip NaN correlations (can happen if a column is constant)
            if pd.isna(corr_value):
                continue
            if abs(corr_value) > threshold:
                pairs.append((columns[i], columns[j], float(corr_value)))

    return pairs


def _build_correlation_finding(
    col_a: str,
    col_b: str,
    corr_value: float,
    threshold: float,
) -> Finding:
    """Create a Finding for a highly correlated column pair.

    Always includes the caveat about Pearson's limitations.
    """
    direction = "positively" if corr_value > 0 else "negatively"

    message = (
        f"Columns '{col_a}' and '{col_b}' are strongly {direction} correlated "
        f"(r={corr_value:.3f}, threshold={threshold}). "
        f"This suggests redundancy or multicollinearity — consider dropping "
        f"one before modelling. "
        f"Note: Pearson correlation only measures linear relationships."
    )

    return Finding(
        column=f"{col_a} & {col_b}",
        check="correlation",
        severity=SEVERITY_HIGH,
        metric=f"r={corr_value:.3f}",
        message=message,
    )


def _build_summary(
    findings: List[Finding], total_numeric: int, threshold: float
) -> str:
    """One-line summary for the report."""
    if not findings:
        return (
            f"No highly correlated pairs found among {total_numeric} "
            f"numeric columns (threshold: |r| > {threshold})."
        )
    return (
        f"Found {len(findings)} highly correlated column pair(s) "
        f"among {total_numeric} numeric columns (threshold: |r| > {threshold})."
    )
