"""
Report generator for dqcheck.

Builds either a Markdown or HTML report from a ReportData object.
The report includes:
  1. A summary header with the health score, severity counts, and
     basic dataset info.
  2. Per-check sections, each listing the findings with severity badges.

The health score formula:
  Start at 100 and deduct points for each finding:
    - High severity:   -10 points each
    - Medium severity:  -3 points each
    - Low severity:     -1 point each
    - Info:              0 points (informational only)
  Minimum score is 0. This is deliberately simple and transparent so
  users can understand and explain it.

The HTML report is fully self-contained (inline CSS, no external
dependencies) so it can be opened in any browser without an internet
connection.
"""

from typing import List

from dqcheck.models import (
    ReportData,
    CheckResult,
    Finding,
    DatasetInfo,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO,
)


# ---------------------------------------------------------------------------
# Health score calculation
# ---------------------------------------------------------------------------


def calculate_health_score(check_results: List[CheckResult]) -> tuple:
    """Compute the health score and severity counts.

    Formula:
      score = 100 - (high_count * 10) - (medium_count * 3) - (low_count * 1)
      Clamped to [0, 100].

    This is intentionally simple. A more sophisticated score could weight
    by column importance or data volume, but simplicity and transparency
    are more important here.

    Returns:
        A tuple of (score, high_count, medium_count, low_count, info_count).
    """
    high_count = 0
    medium_count = 0
    low_count = 0
    info_count = 0

    for cr in check_results:
        for finding in cr.findings:
            if finding.severity == SEVERITY_HIGH:
                high_count += 1
            elif finding.severity == SEVERITY_MEDIUM:
                medium_count += 1
            elif finding.severity == SEVERITY_LOW:
                low_count += 1
            elif finding.severity == SEVERITY_INFO:
                info_count += 1

    score = 100 - (high_count * 10) - (medium_count * 3) - (low_count * 1)
    score = max(0, min(100, score))

    return score, high_count, medium_count, low_count, info_count


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_markdown_report(report: ReportData) -> str:
    """Build a complete Markdown report from the analysis results.

    Args:
        report: The ReportData containing all check results.

    Returns:
        A string of Markdown-formatted text.
    """
    sections = []
    sections.append(_md_header(report))
    sections.append(_md_dataset_info(report.dataset_info))

    for cr in report.check_results:
        sections.append(_md_check_section(cr))

    sections.append(_md_footer())
    return "\n\n".join(sections)


def _md_header(report: ReportData) -> str:
    """Markdown header with health score and severity summary."""
    score = report.health_score
    grade = _score_to_grade(score)

    lines = [
        "# 📊 dqcheck — Data Quality Report",
        "",
        f"**Health Score: {score}/100 ({grade})**",
        "",
        f"| Severity | Count |",
        f"|----------|-------|",
        f"| 🔴 High   | {report.high_count}     |",
        f"| 🟡 Medium | {report.medium_count}     |",
        f"| 🟢 Low    | {report.low_count}     |",
        f"| ℹ️ Info   | {report.info_count}     |",
        "",
        "_Health score formula: start at 100, deduct 10 per high, "
        "3 per medium, 1 per low finding. Min 0._",
    ]
    return "\n".join(lines)


def _md_dataset_info(info: DatasetInfo) -> str:
    """Markdown section with basic dataset metadata."""
    lines = [
        "## Dataset Info",
        "",
        f"- **File:** `{info.file_path}`",
        f"- **Rows:** {info.row_count:,}",
        f"- **Columns:** {info.column_count}",
        "",
        "| Column | Dtype |",
        "|--------|-------|",
    ]

    for col, dtype in info.dtypes.items():
        lines.append(f"| {col} | {dtype} |")

    if info.dtype_warnings:
        lines.append("")
        lines.append("### ⚠️ Dtype Warnings")
        lines.append("")
        for warning in info.dtype_warnings:
            lines.append(
                f"- **{warning.column}**: stored as `{warning.current_dtype}`, "
                f"but looks like `{warning.likely_dtype}`. {warning.evidence}"
            )

    return "\n".join(lines)


def _md_check_section(cr: CheckResult) -> str:
    """Markdown section for a single check's results."""
    lines = [
        f"## {cr.name}",
        "",
        f"_{cr.summary}_",
    ]

    if not cr.findings:
        lines.append("\nNo issues found. ✅")
        return "\n".join(lines)

    lines.append("")
    for finding in cr.findings:
        badge = _severity_badge_md(finding.severity)
        lines.append(f"### {badge} {finding.column}")
        lines.append("")
        lines.append(f"- **Check:** {finding.check}")
        lines.append(f"- **Severity:** {finding.severity}")
        lines.append(f"- **Metric:** {finding.metric}")
        lines.append(f"- **Details:** {finding.message}")
        lines.append("")

    return "\n".join(lines)


def _severity_badge_md(severity: str) -> str:
    """Return an emoji badge for the severity level."""
    badges = {
        SEVERITY_HIGH: "🔴",
        SEVERITY_MEDIUM: "🟡",
        SEVERITY_LOW: "🟢",
        SEVERITY_INFO: "ℹ️",
    }
    return badges.get(severity, "❓")


def _md_footer() -> str:
    """Markdown footer with disclaimers."""
    return (
        "---\n\n"
        "_Generated by dqcheck. "
        "This report flags potential issues — not all findings are errors. "
        "Always apply domain knowledge before making data decisions._"
    )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def build_html_report(report: ReportData) -> str:
    """Build a self-contained HTML report with inline CSS.

    The HTML uses a clean, modern design with colour-coded severity badges.
    No external stylesheets, scripts, or fonts are required.

    Args:
        report: The ReportData containing all check results.

    Returns:
        A string of complete HTML.
    """
    css = _html_css()
    body = _html_body(report)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>dqcheck — Data Quality Report</title>
<style>
{css}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _html_css() -> str:
    """Inline CSS for the HTML report.

    Uses a clean design with:
    - A neutral background and dark text for readability.
    - Colour-coded severity badges (red, amber, green, blue).
    - Card-like sections with subtle shadows.
    """
    return """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        background: #f5f6fa;
        color: #2d3436;
        line-height: 1.6;
        padding: 2rem;
        max-width: 960px;
        margin: 0 auto;
    }
    h1 { font-size: 1.8rem; margin-bottom: 0.5rem; color: #2d3436; }
    h2 { font-size: 1.3rem; margin: 1.5rem 0 0.5rem; color: #2d3436;
         border-bottom: 2px solid #dfe6e9; padding-bottom: 0.3rem; }
    h3 { font-size: 1.05rem; margin: 1rem 0 0.3rem; }
    .card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .score-box {
        display: inline-block;
        font-size: 2.5rem;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0 1rem;
    }
    .score-good { background: #00b894; color: white; }
    .score-ok   { background: #fdcb6e; color: #2d3436; }
    .score-bad  { background: #d63031; color: white; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    .badge-high   { background: #d63031; }
    .badge-medium { background: #e17055; }
    .badge-low    { background: #00b894; }
    .badge-info   { background: #0984e3; }
    .summary-table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
    .summary-table th, .summary-table td {
        padding: 0.4rem 0.8rem;
        text-align: left;
        border-bottom: 1px solid #dfe6e9;
    }
    .summary-table th { background: #dfe6e9; font-weight: 600; }
    .finding-card {
        background: #fafafa;
        border-left: 4px solid #dfe6e9;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
    .finding-card.high   { border-left-color: #d63031; }
    .finding-card.medium { border-left-color: #e17055; }
    .finding-card.low    { border-left-color: #00b894; }
    .finding-card.info   { border-left-color: #0984e3; }
    .meta { color: #636e72; font-size: 0.9rem; }
    .footer { text-align: center; color: #b2bec3; font-size: 0.85rem; margin-top: 2rem; }
    """


def _html_body(report: ReportData) -> str:
    """Build the HTML body content."""
    parts = []
    parts.append(_html_header(report))
    parts.append(_html_dataset_info(report.dataset_info))

    for cr in report.check_results:
        parts.append(_html_check_section(cr))

    parts.append(_html_footer())
    return "\n".join(parts)


def _html_header(report: ReportData) -> str:
    """HTML header card with health score and summary counts."""
    score = report.health_score
    grade = _score_to_grade(score)

    if score >= 70:
        score_class = "score-good"
    elif score >= 40:
        score_class = "score-ok"
    else:
        score_class = "score-bad"

    return f"""
    <div class="card">
        <h1>&#128202; dqcheck — Data Quality Report</h1>
        <div class="score-box {score_class}">{score}/100</div>
        <p><strong>Grade: {grade}</strong></p>
        <table class="summary-table">
            <tr><th>Severity</th><th>Count</th></tr>
            <tr><td><span class="badge badge-high">HIGH</span></td><td>{report.high_count}</td></tr>
            <tr><td><span class="badge badge-medium">MEDIUM</span></td><td>{report.medium_count}</td></tr>
            <tr><td><span class="badge badge-low">LOW</span></td><td>{report.low_count}</td></tr>
            <tr><td><span class="badge badge-info">INFO</span></td><td>{report.info_count}</td></tr>
        </table>
        <p class="meta">Health score formula: start at 100, deduct 10 per high,
        3 per medium, 1 per low finding. Min 0.</p>
    </div>"""


def _html_dataset_info(info: DatasetInfo) -> str:
    """HTML card with dataset metadata."""
    rows = ""
    for col, dtype in info.dtypes.items():
        rows += f"<tr><td>{_html_escape(col)}</td><td>{dtype}</td></tr>\n"

    warnings_html = ""
    if info.dtype_warnings:
        warnings_html = "<h3>⚠️ Dtype Warnings</h3><ul>"
        for w in info.dtype_warnings:
            warnings_html += (
                f"<li><strong>{_html_escape(w.column)}</strong>: "
                f"stored as <code>{w.current_dtype}</code>, "
                f"likely <code>{w.likely_dtype}</code>. "
                f"{_html_escape(w.evidence)}</li>"
            )
        warnings_html += "</ul>"

    return f"""
    <div class="card">
        <h2>Dataset Info</h2>
        <p><strong>File:</strong> <code>{_html_escape(info.file_path)}</code></p>
        <p><strong>Rows:</strong> {info.row_count:,} &nbsp;|&nbsp;
           <strong>Columns:</strong> {info.column_count}</p>
        <table class="summary-table">
            <tr><th>Column</th><th>Dtype</th></tr>
            {rows}
        </table>
        {warnings_html}
    </div>"""


def _html_check_section(cr: CheckResult) -> str:
    """HTML card for one check's results."""
    findings_html = ""

    if not cr.findings:
        findings_html = "<p>No issues found. ✅</p>"
    else:
        for finding in cr.findings:
            badge = _severity_badge_html(finding.severity)
            css_class = finding.severity
            findings_html += f"""
            <div class="finding-card {css_class}">
                <h3>{badge} {_html_escape(finding.column)}</h3>
                <p class="meta">Check: {_html_escape(finding.check)}
                &nbsp;|&nbsp; Metric: {_html_escape(finding.metric)}</p>
                <p>{_html_escape(finding.message)}</p>
            </div>"""

    return f"""
    <div class="card">
        <h2>{_html_escape(cr.name)}</h2>
        <p class="meta">{_html_escape(cr.summary)}</p>
        {findings_html}
    </div>"""


def _severity_badge_html(severity: str) -> str:
    """Return an HTML badge span for the severity level."""
    label = severity.upper()
    return f'<span class="badge badge-{severity}">{label}</span>'


def _html_footer() -> str:
    """HTML footer."""
    return """
    <div class="footer">
        <p>Generated by dqcheck. This report flags potential issues —
        not all findings are errors. Always apply domain knowledge
        before making data decisions.</p>
    </div>"""


def _html_escape(text: str) -> str:
    """Escape special HTML characters to prevent rendering issues.

    We do this manually instead of importing html.escape to keep
    dependencies explicit and visible.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_to_grade(score: int) -> str:
    """Convert a numeric health score to a letter grade.

    90-100 → Excellent
    70-89  → Good
    50-69  → Fair
    30-49  → Poor
    0-29   → Critical
    """
    if score >= 90:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    if score >= 30:
        return "Poor"
    return "Critical"
