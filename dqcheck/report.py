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
        "# dqcheck — Data Quality Report",
        "",
        f"**Health Score: {score}/100 ({grade})**",
        "",
        "| Severity | Count |",
        "|----------|-------|",
        f"| HIGH     | {report.high_count}     |",
        f"| MEDIUM   | {report.medium_count}     |",
        f"| LOW      | {report.low_count}     |",
        f"| INFO     | {report.info_count}     |",
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
        lines.append("### Dtype Warnings")
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
        lines.append("\nNo issues found.")
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
    """Return a plain-text severity badge for Markdown."""
    return f"[{severity.upper()}]"


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
    """Inline CSS for the HTML report."""
    return """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: #f8fafc;
        color: #0f172a;
        line-height: 1.5;
        padding: 2.5rem 1.5rem;
        max-width: 960px;
        margin: 0 auto;
    }
    h1 { font-size: 1.75rem; font-weight: 700; color: #0f172a; letter-spacing: -0.025em; }
    h2 { font-size: 1.25rem; font-weight: 600; margin: 1.25rem 0 0.5rem; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.35rem; }
    h3 { font-size: 1rem; font-weight: 600; margin: 0.75rem 0 0.25rem; color: #1e293b; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; background: #f1f5f9; padding: 0.15rem 0.35rem; border-radius: 4px; font-size: 0.875rem; color: #0f172a; }
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .score-container {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        margin: 1rem 0;
    }
    .score-box {
        font-size: 2.25rem;
        font-weight: 800;
        padding: 0.35rem 1.25rem;
        border-radius: 8px;
        letter-spacing: -0.025em;
    }
    .score-good { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .score-ok   { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .score-bad  { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .score-meta { font-size: 1.1rem; color: #334155; }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    .badge-high   { background: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
    .badge-medium { background: #fef3c7; color: #92400e; border: 1px solid #fbbf24; }
    .badge-low    { background: #dcfce7; color: #166534; border: 1px solid #4ade80; }
    .badge-info   { background: #e0f2fe; color: #075985; border: 1px solid #7dd3fc; }
    .summary-table { width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.9rem; }
    .summary-table th, .summary-table td {
        padding: 0.5rem 0.75rem;
        text-align: left;
        border-bottom: 1px solid #f1f5f9;
    }
    .summary-table th { background: #f8fafc; font-weight: 600; color: #475569; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #cbd5e1; }
    .finding-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #94a3b8;
        padding: 0.85rem 1rem;
        margin: 0.6rem 0;
        border-radius: 0 6px 6px 0;
    }
    .finding-card.high   { border-left-color: #ef4444; background: #fffdfd; }
    .finding-card.medium { border-left-color: #f59e0b; background: #fffdfa; }
    .finding-card.low    { border-left-color: #10b981; background: #fafdfb; }
    .finding-card.info   { border-left-color: #3b82f6; background: #fafdff; }
    .meta { color: #64748b; font-size: 0.875rem; margin-top: 0.25rem; }
    .footer { text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; }
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
        <h1>dqcheck — Data Quality Report</h1>
        <div class="score-container">
            <div class="score-box {score_class}">{score}/100</div>
            <div class="score-meta">
                <strong>Rating: {grade}</strong>
                <p class="meta">Health score formula: start at 100, deduct 10 per high, 3 per medium, 1 per low finding. Min 0.</p>
            </div>
        </div>
        <table class="summary-table">
            <thead>
                <tr><th>Severity</th><th>Findings Count</th></tr>
            </thead>
            <tbody>
                <tr><td><span class="badge badge-high">HIGH</span></td><td>{report.high_count}</td></tr>
                <tr><td><span class="badge badge-medium">MEDIUM</span></td><td>{report.medium_count}</td></tr>
                <tr><td><span class="badge badge-low">LOW</span></td><td>{report.low_count}</td></tr>
                <tr><td><span class="badge badge-info">INFO</span></td><td>{report.info_count}</td></tr>
            </tbody>
        </table>
    </div>"""


def _html_dataset_info(info: DatasetInfo) -> str:
    """HTML card with dataset metadata."""
    rows = ""
    for col, dtype in info.dtypes.items():
        rows += f"<tr><td><code>{_html_escape(col)}</code></td><td><code>{dtype}</code></td></tr>\n"

    warnings_html = ""
    if info.dtype_warnings:
        warnings_html = "<h3 style='margin-top:1rem;color:#b45309;'>Dtype Warnings</h3><ul>"
        for w in info.dtype_warnings:
            warnings_html += (
                f"<li style='margin:0.25rem 0;'><code>{_html_escape(w.column)}</code>: "
                f"stored as <code>{w.current_dtype}</code>, "
                f"likely <code>{w.likely_dtype}</code>. "
                f"{_html_escape(w.evidence)}</li>"
            )
        warnings_html += "</ul>"

    return f"""
    <div class="card">
        <h2>Dataset Info</h2>
        <p class="meta" style="margin-bottom:0.75rem;">
            <strong>Target:</strong> <code>{_html_escape(info.file_path)}</code> &nbsp;|&nbsp;
            <strong>Rows:</strong> {info.row_count:,} &nbsp;|&nbsp;
            <strong>Columns:</strong> {info.column_count}
        </p>
        <table class="summary-table">
            <thead>
                <tr><th>Column Name</th><th>Detected Dtype</th></tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        {warnings_html}
    </div>"""


def _html_check_section(cr: CheckResult) -> str:
    """HTML card for one check's results."""
    findings_html = ""

    if not cr.findings:
        findings_html = "<p style='color:#16a34a;font-size:0.9rem;margin-top:0.5rem;'>No issues detected for this check.</p>"
    else:
        for finding in cr.findings:
            badge = _severity_badge_html(finding.severity)
            css_class = finding.severity
            findings_html += f"""
            <div class="finding-card {css_class}">
                <h3>{badge} {_html_escape(finding.column)}</h3>
                <p class="meta">Check: <strong>{_html_escape(finding.check)}</strong>
                &nbsp;|&nbsp; Metric: <code>{_html_escape(finding.metric)}</code></p>
                <p style="margin-top:0.4rem;">{_html_escape(finding.message)}</p>
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
