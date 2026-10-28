# dqcheck — CSV Data Quality Checker

**"How much can I trust this data before building a model?"**

dqcheck is a lightweight, readable command-line tool that analyses a CSV file and produces a data quality report. It checks for missing values (including hidden ones), duplicates, outliers, skewness, high correlations, constant columns, and ID-like columns — then gives you a transparent health score out of 100.

## Why dqcheck?

| Feature | dqcheck | ydata-profiling | sweetviz |
|---------|---------|-----------------|----------|
| Dependencies | 3 (pandas, numpy, pytest) | 30+ | 15+ |
| Output | Markdown or HTML | HTML | HTML |
| Install time | Seconds | Minutes | Minutes |
| Code you can read | ✅ Every function has a docstring | ❌ Complex codebase | ❌ Complex codebase |
| Modifies your data | Never | Never | Never |
| Health score | ✅ Transparent formula | ❌ No single score | ❌ No single score |

**dqcheck is designed to be read.** Every function has a docstring explaining *what* it does and *why* the approach was chosen. A data science student should be able to open any module and understand the statistics.

## Install

```bash
# Clone and install dependencies
git clone <repo-url>
cd dqcheck
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage — prints Markdown report to stdout
python -m dqcheck data.csv

# With key-duplicate checking
python -m dqcheck data.csv --key id,date

# HTML report saved to a file
python -m dqcheck data.csv --format html --output report.html

# Z-score outlier method instead of IQR
python -m dqcheck data.csv --outlier-method zscore

# Custom correlation threshold
python -m dqcheck data.csv --corr-threshold 0.8

# Extra hidden-missing tokens
python -m dqcheck data.csv --missing-tokens "?,-, unknown"
```

### All options

```
python -m dqcheck --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `csv_file` | (required) | Path to the CSV file |
| `--key` | None | Comma-separated key columns for duplicate checking |
| `--format` | `md` | Output format: `md` or `html` |
| `--output` | stdout | Write report to this file path |
| `--outlier-method` | `iqr` | Outlier method: `iqr` or `zscore` |
| `--corr-threshold` | `0.9` | Absolute correlation threshold for flagging |
| `--missing-tokens` | built-in set | Extra tokens to treat as missing values |

## Example output (excerpt)

```
# 📊 dqcheck — Data Quality Report

**Health Score: 72/100 (Good)**

| Severity | Count |
|----------|-------|
| 🔴 High   | 2     |
| 🟡 Medium | 3     |
| 🟢 Low    | 5     |
| ℹ️ Info   | 1     |

## Missing Values
_Found missing values in 4/12 columns (1 high, 1 medium, 2 low severity)._

### 🔴 notes
- **Check:** missing
- **Severity:** high
- **Metric:** 65/200 (32.5%)
- **Details:** Column 'notes' has 65 missing values (32.5% of 200 rows).
  45 are standard NaN/None, and 20 are hidden missing values
  (sentinel strings like '?', 'N/A', etc.).
  Consider dropping this column — over half the values are missing.
```

## Try it on the sample data

```bash
python -m dqcheck sample_data/messy_sample.csv --key customer_id --format html --output report.html
```

Then open `report.html` in your browser.

## Run the tests

```bash
pytest tests/ -v
```

## Design decisions

### Why IQR as the default outlier method?
The IQR method is **robust to skewed distributions** and does not assume normality. Real-world data is rarely normally distributed, so IQR gives fewer false positives than z-score. Z-score is available via `--outlier-method zscore` for cases where you know the data is roughly normal.

### Why flag outliers instead of removing them?
An outlier is not necessarily an error. A $500,000 salary in a dataset of salaries is unusual but might be a real CEO salary. **dqcheck never modifies your data.** It only reports what it finds and leaves the decision to you.

### Health score formula
```
score = 100 - (high_count × 10) - (medium_count × 3) - (low_count × 1)
```
Clamped to [0, 100]. This is intentionally simple and transparent. You can explain it to a stakeholder in one sentence.

### Why not ydata-profiling or sweetviz?
Those are excellent tools for exploratory data analysis, but they:
- Pull in dozens of dependencies (matplotlib, scipy, seaborn, etc.)
- Generate large, complex HTML reports
- Have codebases that are hard to read and modify
- Don't give you a single "trustworthiness" score

dqcheck is for when you want a **quick, lightweight, readable** quality check before modelling — not a full EDA report.

## Limitations

- **Only CSV files.** No Parquet, Excel, or database connections.
- **Pearson correlation only.** Non-linear relationships are not detected.
- **No visualization.** Use pandas/matplotlib for plots after reviewing the report.
- **Heuristic dtype detection.** The "likely wrong dtype" check uses regex patterns and may miss unusual formats.
- **Health score is simple.** It does not weight columns by importance or adjust for dataset size.

## Project structure

```
dqcheck/
├── dqcheck/
│   ├── __init__.py          # Package init
│   ├── __main__.py          # python -m dqcheck entry point
│   ├── cli.py               # Argument parsing and pipeline orchestration
│   ├── loader.py            # CSV loading and dtype inspection
│   ├── models.py            # Dataclasses for findings and reports
│   ├── report.py            # Markdown and HTML report generation
│   └── checks/
│       ├── __init__.py
│       ├── missing.py       # Missing and hidden-missing values
│       ├── duplicates.py    # Full-row and key duplicates
│       ├── outliers.py      # IQR and z-score outlier detection
│       ├── skewness.py      # Distribution skewness
│       ├── correlation.py   # Pearson correlation pairs
│       └── structure.py     # Constant and ID-like columns
├── tests/                   # One test file per module + end-to-end
├── sample_data/
│   └── messy_sample.csv     # Deliberately dirty dataset (~200 rows)
├── docs/
│   └── LOGIC.md             # Statistics explainer for each check
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## License

MIT — see [LICENSE](LICENSE).
