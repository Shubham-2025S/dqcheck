"""
Data models for dqcheck findings and reports.

Uses dataclasses to provide clear, typed containers for every piece of
information the tool produces. Each Finding represents one issue found
in a single column (or a pair of columns). CheckResult groups findings
from a single check module. ReportData holds the full analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Severity levels as plain strings so the code stays simple and grep-able.
# ---------------------------------------------------------------------------
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_INFO = "info"


@dataclass
class Finding:
    """A single quality issue found in the data.

    Attributes:
        column:   The column name this finding applies to.
                  For row-level checks (e.g. duplicates), this may be empty.
        check:    Which check produced it (e.g. "missing", "outlier").
        severity: One of "low", "medium", "high", or "info".
        metric:   A short numeric summary (e.g. "12.5%").
        message:  A plain-English explanation of the issue and why it matters.
    """

    column: str
    check: str
    severity: str
    metric: str
    message: str


@dataclass
class DtypeWarning:
    """A warning about a column whose inferred dtype is likely wrong.

    For example, a column stored as 'object' that contains values like
    "$1,200" or "2024-01-15" — it looks numeric or date-like but pandas
    could not parse it automatically.

    Attributes:
        column:        The column name.
        current_dtype: The dtype pandas assigned (usually 'object').
        likely_dtype:  What the column probably should be ('numeric' or 'date').
        evidence:      Example values that led to this conclusion.
    """

    column: str
    current_dtype: str
    likely_dtype: str
    evidence: str


@dataclass
class DatasetInfo:
    """Basic metadata about the loaded CSV file.

    Attributes:
        file_path:      The path to the CSV file.
        row_count:      Number of data rows (excluding the header).
        column_count:   Number of columns.
        dtypes:         A dict mapping column name → pandas dtype string.
        dtype_warnings: Columns whose dtype is probably wrong.
    """

    file_path: str
    row_count: int
    column_count: int
    dtypes: dict
    dtype_warnings: List[DtypeWarning] = field(default_factory=list)


@dataclass
class CheckResult:
    """The output of a single check module.

    Attributes:
        name:     Human-readable name (e.g. "Missing Values").
        findings: List of individual findings produced by this check.
        summary:  A one-line summary sentence for the report header.
    """

    name: str
    findings: List[Finding] = field(default_factory=list)
    summary: str = ""


@dataclass
class ReportData:
    """Everything needed to build the final report.

    Attributes:
        dataset_info:  Metadata about the CSV file.
        check_results: Ordered list of check results.
        health_score:  An overall score from 0 to 100.
        high_count:    Number of high-severity findings.
        medium_count:  Number of medium-severity findings.
        low_count:     Number of low-severity findings.
        info_count:    Number of info-level findings.
    """

    dataset_info: DatasetInfo
    check_results: List[CheckResult] = field(default_factory=list)
    health_score: int = 100
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
