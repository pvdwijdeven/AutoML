import pandas as pd
import re
from collections import Counter, defaultdict
from scipy.stats import skew
from automl_libs import infer_dtype


def analyze_target(df, target_col):
    suggestions = []
    target = df[target_col]
    target_non_null = target.dropna()

    # Basic info
    n_unique = target_non_null.nunique()
    has_missing = target.isnull().any()

    # Detect type: numeric or categorical
    if infer_dtype(target) == "float" or infer_dtype(target) == "integer":
        # Regression or numeric target

        # Missing values
        if has_missing:
            suggestions.append(
                "Target has missing values — consider imputing or excluding those rows"
            )

        # Low variance
        top_freq = target_non_null.value_counts(normalize=True).max()
        if top_freq > 0.95:
            suggestions.append(
                "Target has low variance — may be difficult to model"
            )

        # Skewness (only for continuous numeric)
        if n_unique > 10:
            sk = skew(target_non_null)
            if abs(sk) > 1:
                suggestions.append(
                    f"Target is highly skewed (skew={sk:.2f}) — consider log or Box-Cox transform"
                )

        # Outliers
        if len(target_non_null) >= 10:
            q1 = target_non_null.quantile(0.25)
            q3 = target_non_null.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = target_non_null[
                (target_non_null < lower) | (target_non_null > upper)
            ]
            outlier_ratio = len(outliers) / len(target_non_null)
            if outlier_ratio > 0.02:
                suggestions.append(
                    f"Target has {outlier_ratio:.1%} outliers — consider robust loss functions or outlier capping"
                )

        # Multimodality (rough check: count modes)
        counts = Counter(target_non_null)
        if (
            n_unique > 10
            and len(
                [c for c in counts.values() if c > len(target_non_null) * 0.1]
            )
            > 1
        ):
            suggestions.append(
                "Target distribution appears multimodal — consider modeling subgroups separately"
            )

    else:
        # Classification or categorical target

        # Missing values
        if has_missing:
            suggestions.append(
                "Target has missing values — consider imputing or excluding those rows"
            )

        # Check imbalance
        value_counts = target_non_null.value_counts(normalize=True)
        max_ratio = value_counts.max()
        if max_ratio > 0.8:
            suggestions.append(
                f"Target is imbalanced (dominant class {max_ratio:.0%}) — consider resampling or class weights"
            )

        # High cardinality
        if n_unique > 20:
            suggestions.append(
                f"Target has high cardinality ({n_unique} classes) — consider grouping rare classes"
            )

        # Low variance (only one class)
        if n_unique == 1:
            suggestions.append(
                "Target has only one class — no variation to learn from"
            )

    return suggestions


def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), len(series), outliers


def analyze_boolean_column(
    df: pd.DataFrame, column_name: str, sample_size=1000
):

    # Reduce size for large datasets
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    col_data = df[column_name]
    col_non_null = col_data.dropna()
    true_ratio = col_non_null.mean()

    suggestions = ["Consider replacing with binary encoding (0/1)"]

    if col_data.nunique(dropna=True) == 1:
        suggestions.append("Drop constant boolean column")

    if true_ratio < 0.05 or true_ratio > 0.95:
        suggestions.append("Class imbalance (may need resampling or weight)")

    return suggestions


def analyze_numeric_column(
    df: pd.DataFrame, column_name: str, sample_size=10000
) -> list[str]:
    suggestions = []

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    col_data = df[column_name]
    col_type = col_data.dtype
    col_non_null = col_data.dropna()

    info = {
        "dtype": str(col_type),
        "n_unique": col_non_null.nunique(),
        "has_missing": col_data.isnull().any(),
        "min": col_non_null.min() if not col_non_null.empty else None,
        "max": col_non_null.max() if not col_non_null.empty else None,
        "skewness": skew(col_non_null) if len(col_non_null) > 2 else None,
    }

    if info["n_unique"] == 1:
        suggestions.append("Drop constant column")
    elif info["n_unique"] == 2:
        suggestions.append("Consider treating as binary (2 unique values)")

    if info["n_unique"] < 10 and col_type == "int64":
        suggestions.append("Treat as categorical (low-cardinality integer)")

    if info["skewness"] is not None and abs(info["skewness"]) > 1:
        suggestions.append("Try log or Box-Cox transform (high skew)")

    # INTEGER-SPECIFIC
    if infer_dtype(col_data) == "integer":
        suggestions.append(
            "Use frequency encoding or label encoding (integer type)"
        )
        if info["n_unique"] > 20:
            suggestions.append(
                "Consider binning (bucketization) (integer with many unique values)"
            )

    # FLOAT-SPECIFIC
    elif infer_dtype(col_data) == "float":
        suggestions.append("Scale values (e.g. StandardScaler/MinMax)")
        if info["min"] is not None and info["min"] >= 0:
            suggestions.append("Try log transform (positive-only float)")

    # Temporal hints
    if column_name.lower().endswith(("year", "month", "day", "hour")):
        suggestions.append("Extract date parts or treat as temporal")

    # Compute some shared stats
    non_zero_values = col_non_null[col_non_null != 0]
    zero_ratio = 1 - (len(non_zero_values) / len(col_non_null))

    # Check for zero-inflated + skewed + outliers
    if zero_ratio > 0.5 and len(non_zero_values) > 10:
        skewness_non_zero = skew(non_zero_values)
        n_outliers, _, _ = detect_outliers_iqr(non_zero_values)

        if abs(skewness_non_zero) > 1 and n_outliers > 0:
            suggestions.append(
                "Zero-inflated + skewed with outliers — consider splitting into binary 'has_value' + log1p-transformed non-zero part"
            )
    if len(col_non_null) >= 10:  # minimum samples for IQR
        n_outliers, total, _ = detect_outliers_iqr(col_non_null)
        outlier_ratio = n_outliers / total
        if outlier_ratio > 0.02:  # over 2% flagged as outliers
            suggestions.append(
                f"{outlier_ratio:.1%} outliers detected — consider capping or robust scaling"
            )

    return suggestions


def analyze_string_column(
    df: pd.DataFrame,
    column_name: str,
    top_n=5,
    max_prefix_suffix_len=5,
    delimiter_consistency_threshold=0.8,
) -> list[str]:
    """
    Analyzes a string column and returns feature engineering suggestions.
    """
    suggestions = []
    series = df[column_name]
    series = series.dropna().astype(str)

    if series.empty:
        return ["Series is empty after removing nulls."]

    # --- 1. String Length ---
    str_lengths = series.str.len()
    if str_lengths.nunique() > 1 and str_lengths.std() > 0.5:
        suggestions.append("Use string length as a feature.")

    # --- 2. Word Count ---
    word_counts = series.str.split().apply(len)
    if word_counts.nunique() > 1 and word_counts.std() > 0.5:
        suggestions.append("Use word count as a feature.")

    # --- 3. Delimiter Splitting ---
    delimiter_regex = re.compile(r"[-_/:|]")
    split_positions = defaultdict(list)

    for s in series:
        for match in delimiter_regex.finditer(s):
            split_positions[match.group()].append(match.start())

    for delim, positions in split_positions.items():
        if not positions:
            continue
        most_common_pos = Counter(positions).most_common(1)[0]
        freq = most_common_pos[1] / len(series)
        if freq >= delimiter_consistency_threshold:
            suggestions.append(
                f"Split on '{delim}' at position {most_common_pos[0]} (consistent in {freq:.0%} of samples)."
            )

    # --- 4. Contains Digits / Alpha / Alnum ---
    if series.str.contains(r"\d").mean() > 0.5:
        suggestions.append("Extract numeric parts from strings.")

    if series.str.fullmatch(r"[A-Za-z]+", na=False).mean() > 0.5:
        suggestions.append(
            "Alphabetic strings; consider character-level analysis."
        )
    elif series.str.fullmatch(r"[A-Za-z0-9]+", na=False).mean() > 0.5:
        suggestions.append(
            "Alphanumeric strings; consider splitting letters and digits."
        )

    # --- 5. Prefix / Suffix Analysis ---
    for n in range(1, max_prefix_suffix_len + 1):
        # Prefix
        prefix_counts = Counter(s[:n] for s in series if len(s) >= n)
        if not prefix_counts:
            continue
        p_common = prefix_counts.most_common(1)[0]
        if p_common[1] / len(series) >= 0.1:
            suggestions.append(
                f"Prefix length {n}: '{p_common[0]}' in {p_common[1]/len(series):.0%} of samples."
            )

        # Suffix
        suffix_counts = Counter(s[-n:] for s in series if len(s) >= n)
        s_common = suffix_counts.most_common(1)[0]
        if s_common[1] / len(series) >= 0.1:
            suggestions.append(
                f"Suffix length {n}: '{s_common[0]}' in {s_common[1]/len(series):.0%} of samples."
            )

    # --- 6. Top Frequent Values ---
    common = series.value_counts().head(top_n)
    if common.iloc[0] / len(series) > 0.05:
        suggestions.append(
            f"Top {top_n} values appear frequently; consider one-hot or target encoding."
        )

    return suggestions


def analyze_categorical_column(df: pd.DataFrame, column_name: str) -> list[str]:
    suggestions = []

    s = df[column_name]
    n_unique = s.nunique(dropna=True)
    freq = s.value_counts(normalize=True, dropna=True)
    top_freq = freq.iloc[0] if not freq.empty else 0

    suggestions = []

    # Cardinality-based suggestions
    if n_unique == 1:
        suggestions.append(
            f"Column '{column_name}' is constant. Consider dropping it."
        )
    elif n_unique <= 5:
        suggestions.append(
            f"Column '{column_name}' has low cardinality ({n_unique}). Consider One-Hot Encoding."
        )
        suggestions.append(
            "Check if categories can be ordinal; if yes, try Ordinal Encoding."
        )
    elif n_unique <= 20:
        suggestions.append(
            f"Column '{column_name}' has medium cardinality ({n_unique}). Consider Target Encoding or Frequency Encoding."
        )
    else:
        suggestions.append(
            f"Column '{column_name}' has high cardinality ({n_unique}). Consider embedding or hashing techniques."
        )

    # Frequency distribution
    if top_freq > 0.9:
        suggestions.append(
            f"Column '{column_name}' is dominated by a single category ({top_freq:.1%} frequency). Consider grouping rare categories."
        )

    return suggestions
