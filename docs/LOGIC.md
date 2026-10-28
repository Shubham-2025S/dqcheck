# LOGIC.md — Statistics and Reasoning Behind Each Check

This document explains the statistics and reasoning behind each check in dqcheck. It is written for a data science student who wants to understand *why* each check exists and *how* the numbers are calculated.

---

## 1. Missing Values

### What we check
We count how many values in each column are missing. Missing values come in two forms:

1. **Standard missing:** `NaN`, `None`, or empty cells that pandas recognises automatically.
2. **Hidden missing:** Sentinel strings that *look* like data but actually mean "no value." Common examples: `"?"`, `"-"`, `"N/A"`, `"null"`, `"None"`, `"nan"`, `""` (empty string), `" "` (whitespace only).

### Why hidden missing values matter
Many CSV files use placeholder strings instead of leaving cells empty. If you pass `"N/A"` to a model as a category, it will treat it as a real value — which corrupts your results. For numeric columns, the presence of these strings forces pandas to read the whole column as `object` (string), which prevents any numeric analysis.

### How we detect them
For each non-NaN value, we strip whitespace, convert to lowercase, and compare against a set of known sentinel tokens. Users can add their own tokens via `--missing-tokens`.

### Severity thresholds
| Percentage missing | Severity | Reasoning |
|--------------------|----------|-----------|
| < 5% | Low | A small amount of missing data is normal. Simple imputation (mean, median, mode) usually works. |
| 5% – 30% | Medium | Enough to potentially bias your model. Investigate the *reason* for missingness (MCAR, MAR, MNAR). |
| > 30% | High | The column may be unreliable. Imputation becomes increasingly speculative. |
| > 50% | High + drop recommendation | Over half the values are guesses if you impute. Consider dropping the column entirely. |

---

## 2. Duplicate Rows

### What we check
1. **Full-row duplicates:** Rows where every single column has the same value as another row.
2. **Key duplicates:** Rows that share the same value(s) in user-specified key columns (e.g. `id`, `date`) but differ in other columns.

### Why duplicates matter
- **Full-row duplicates** inflate your dataset and give some observations more weight during training. A model trained on duplicated data will be biased toward those observations.
- **Key duplicates** are more dangerous. If two rows share a primary key but disagree on values, at least one of them is wrong. This suggests a data-entry error, a failed join, or a race condition during ETL.

### How we detect them
We use pandas `.duplicated()` which marks the second and subsequent occurrences. For key duplicates, we group by the key columns and check if any non-key column has more than one unique value within a group.

### Severity
- Full-row duplicates: < 1% low, 1–5% medium, > 5% high.
- Key duplicates with conflicting values: always high (data integrity issue).
- Key duplicates with identical values: low (just full duplicates on that key).

---

## 3. Outliers

### What they are
An outlier is a data point that is far from the rest of the distribution. Outliers can be:
- **Errors:** A typo (age = 400 instead of 40).
- **Legitimate extreme values:** A CEO's salary in a dataset of employee salaries.
- **Interesting signals:** Fraud detection depends on outliers.

**dqcheck only flags outliers. It never removes them.** The decision to keep, transform, or remove an outlier requires domain knowledge.

### IQR method (default)
The Interquartile Range (IQR) is the range between the 25th percentile (Q1) and the 75th percentile (Q3):

```
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```

Any value below the lower bound or above the upper bound is flagged.

**Why IQR is the default:**
- It does not assume the data is normally distributed (most real data isn't).
- It uses percentiles, which are robust to extreme values.
- It is the same logic used in box-and-whisker plots, so it is familiar.

**When IQR fails:** If Q1 == Q3 (many values are the same), then IQR = 0 and the method cannot define meaningful bounds. We skip the column with an info note.

### Z-score method (optional)
The z-score measures how many standard deviations a value is from the mean:

```
z = (x - mean) / std
```

We flag values with |z| > 3 (more than 3 standard deviations away).

**Why z-score is optional, not default:**
- It assumes the data is roughly normally distributed. If the data is skewed, the mean and standard deviation are pulled toward extreme values, which can hide real outliers or flag normal values.
- It is influenced by the very outliers it is trying to detect (the mean changes when you add an outlier).

**When z-score fails:** If std = 0 (all values are identical), division by zero would occur. We skip the column with an info note.

### Severity
- < 1% of values are outliers: low.
- 1–5%: medium.
- > 5%: high (unusual — might indicate a data generation problem).

---

## 4. Skewness

### What it measures
Skewness measures the asymmetry of a distribution. A perfectly symmetric distribution (like the normal distribution) has skewness = 0.

- **Positive skew (right-skewed):** The tail extends to the right. Most values are on the left, with a few very large values pulling the mean up. Example: income data.
- **Negative skew (left-skewed):** The tail extends to the left. Most values are on the right, with a few very small values pulling the mean down. Example: exam scores with a difficult test.

### Why skewness matters for modelling
Many algorithms assume roughly symmetric features:
- **Linear regression:** Skewed features violate the assumption of normally distributed residuals.
- **k-means clustering:** The mean (centroid) is pulled toward the tail, distorting clusters.
- **Neural networks:** Skewed inputs can slow convergence.

### How we measure it
We use the Fisher–Pearson coefficient of skewness (pandas `.skew()`). We also report the mean-vs-median gap: if `mean >> median`, the distribution is right-skewed; if `mean << median`, it is left-skewed.

### Thresholds
| |skew| | Severity | Interpretation |
|--------|----------|----------------|
| < 0.5 | Not flagged | Approximately symmetric |
| 0.5 – 1 | Medium | Moderately skewed |
| > 1 | High | Strongly skewed; consider a transform |

### When we suggest a log transform
A log transform (`log(x)`) compresses large values and spreads out small values, reducing right skew. But `log(0)` is undefined and `log(negative)` is complex, so **we only suggest log transforms when all values are strictly positive.** Otherwise we mention alternatives like square root (for non-negative data) or Yeo–Johnson (which handles any real number).

---

## 5. Correlation

### What we measure
We compute the Pearson correlation coefficient (r) between every pair of numeric columns. Pearson's r measures the strength and direction of the **linear** relationship between two variables:

```
r = cov(X, Y) / (std(X) × std(Y))
```

Values range from -1 (perfect negative linear relationship) to +1 (perfect positive linear relationship). r = 0 means no linear relationship.

### Why high correlation matters
When two features are highly correlated (|r| > 0.9 by default):
- **Multicollinearity** in linear models: Coefficients become unstable and hard to interpret. A small change in the data can flip the sign of a coefficient.
- **Redundancy:** Both features carry nearly the same information. Including both wastes computation without adding predictive power.
- **Feature importance:** Feature-importance scores (e.g. from random forests) are split between correlated features, making both appear less important than they really are.

### Important limitation
**Pearson correlation only measures linear relationships.** Two variables can have a perfect non-linear relationship (e.g. y = x²) and still have r ≈ 0. For non-linear relationships, consider mutual information, Spearman's rank correlation, or visual inspection.

### Why we report each pair once
The correlation matrix is symmetric: corr(A, B) = corr(B, A). We only look at the upper triangle to avoid reporting the same pair twice.

### Default threshold
We flag pairs with |r| > 0.9, which is a commonly used threshold in practice. Users can adjust this with `--corr-threshold`.

---

## 6. Structure (Constant and ID-like Columns)

### Constant columns
A column with 0 or 1 unique values carries **zero information**. It cannot help a model distinguish between observations because every row has the same value (or is empty). Examples: a "country" column that is always "US", a "version" column that is always "1.0".

**These should always be dropped** before modelling. They add noise without signal.

### ID-like columns
An ID-like column has nearly as many unique values as there are rows (≥ 90%) AND has a name that suggests it is an identifier ("id", "uuid", "index", "key", "code", etc.).

**Why IDs should not be features:** A model cannot generalise from unique identifiers. If you train a random forest with "customer_id" as a feature, it will memorise which IDs correspond to which outcomes, but this knowledge is useless for new customers. In extreme cases, it can cause severe overfitting (perfect training accuracy, terrible test accuracy).

ID columns are useful for **joins and lookups**, not for modelling.

---

## 7. Health Score

### Formula
```
score = 100 - (high_count × 10) - (medium_count × 3) - (low_count × 1)
```
Clamped to [0, 100].

### Grades
| Score | Grade |
|-------|-------|
| 90–100 | Excellent |
| 70–89 | Good |
| 50–69 | Fair |
| 30–49 | Poor |
| 0–29 | Critical |

### Why this formula?
We chose the simplest formula that gives useful signal:
- **High issues are 10× worse than low issues.** A high-severity finding (e.g. 40% missing data in a column) is genuinely dangerous for modelling, while a low-severity finding (e.g. 2% missing) is usually manageable.
- **No weighting by column importance.** We do not know which columns are your target, which are features, and which are irrelevant. A more sophisticated score could weight by column importance, but that requires domain knowledge we do not have.
- **Transparency over sophistication.** You can explain this formula to a stakeholder in one sentence. A complex formula would be harder to trust and harder to debug.

### What the score does NOT tell you
- It does not measure data *correctness*. A dataset can score 100/100 and still contain systematically wrong values.
- It does not measure data *completeness*. You might be missing entire important columns.
- It does not guarantee model performance. A "Good" dataset can still produce a bad model if the features are irrelevant to the target.

The health score is a **quick sanity check**, not a guarantee.
