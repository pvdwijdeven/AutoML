import pandas as pd
import re
from collections import Counter
from scipy.stats import skew
from automl_libs import infer_dtype
import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder
import warnings
from typing import List
from scipy.stats import entropy as scipy_entropy
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import matplotlib as mpl

mpl.rcParams.update(
    {
        "text.color": "tab:blue",
        "axes.labelcolor": "tab:blue",
        "xtick.color": "tab:blue",
        "ytick.color": "tab:blue",
        "axes.edgecolor": "tab:blue",
        "axes.titlecolor": "tab:blue",
    }
)


def select_features_by_missingness(
    df: pd.DataFrame,
    target: str = "",
    candidate_features: List[str] = [],
    max_features: int = 100,
    min_row_fraction: float = 0.2,
) -> List[str]:
    """
    Select features sorted by ascending missing values, adding features
    until max_features is reached or dropping NA rows keeps at least min_row_fraction of data.

    Parameters:
    - df: DataFrame with your data
    - candidate_features: list of feature column names to consider
    - max_features: max number of features to select
    - min_row_fraction: minimal fraction of rows to keep after dropping NA

    Returns:
    - selected_features: list of selected features satisfying conditions
    """
    if not candidate_features:
        # If no specific features provided, use all columns
        candidate_features = df.columns.tolist()
    if target and target in candidate_features:
        candidate_features = [f for f in candidate_features if f != target]
    # Sort features by missing values ascending
    missing_counts = df[candidate_features].isna().sum().sort_values()

    selected_features = []
    original_row_count = len(df)

    for feature in missing_counts.index:
        selected_features.append(feature)

        # Calculate how many rows remain after dropping NA in current features
        rows_remaining = df.dropna(subset=selected_features).shape[0]
        fraction_remaining = rows_remaining / original_row_count

        # Check stop conditions
        if (
            len(selected_features) >= max_features
            or fraction_remaining <= min_row_fraction
        ):
            # If fraction dropped below threshold after adding last feature,
            # remove that feature to keep fraction above min_row_fraction
            if fraction_remaining < min_row_fraction:
                selected_features.pop()
            break

    return selected_features


def generate_feature_relations(
    df,
    target="",
    dict_descriptors={},
    max_features=100,
    max_samples=10000,
    logger=None,
):

    warnings.filterwarnings("ignore")

    # Step 1: Select features and target
    features = select_features_by_missingness(df, "")
    if target and target not in features:
        features.append(target)
    num_features = len(features)

    # Step 2: Drop missing values only for selected features
    df = df[features].dropna()

    # Step 3: Sample for performance
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    if target == "":
        target = None

    # Step 4: Identify numeric and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Step 5: Label encode categorical features
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Step 6: Restrict to max_features
    all_features = num_cols + cat_cols
    if len(all_features) > max_features:
        all_features = all_features[:max_features]

    # Step 7: Correlation
    if logger:
        logger.info("[GREEN]  - correlation information.")
    corr_matrix = df[all_features].corr()

    # Step 8: Mutual Information
    if logger:
        logger.info("[GREEN]  - mutual information.")
    mi_scores = {}
    try:
        X_features = [f for f in all_features if f != target]
        if target and target in df.columns:
            y = df[target]
            X = df[X_features]
            if y.nunique() <= 10:
                mi = mutual_info_classif(
                    X, y, discrete_features="auto", random_state=42
                )
            else:
                mi = mutual_info_regression(
                    X, y, discrete_features="auto", random_state=42
                )
            mi_scores = dict(zip(X_features, mi))
    except Exception as e:
        print(f"Error computing mutual information: {e}")
        mi_scores = {feature: None for feature in all_features}

    # Step 9: Aggregate results
    if logger:
        logger.info("[GREEN]- Creating relation info per feature.")
    insights = {}
    for feature in all_features:
        related = []
        suggestions = []

        # Related by correlation
        if feature in corr_matrix:
            high_corr = corr_matrix[feature][
                (corr_matrix[feature].abs() > 0.7)
                & (corr_matrix[feature].abs() < 1.0)
            ]
            related = high_corr.index.tolist()
            if not high_corr.empty:
                suggestions.append(
                    "Highly correlated with "
                    + ", ".join(related)
                    + " — consider dropping or combining."
                )

        # Mutual information
        mi_score = mi_scores.get(feature)
        if mi_score is not None:
            if mi_score > 0.2:
                suggestions.append(
                    f"High mutual information with target (MI={mi_score:.2f}) — useful feature."
                )
            elif mi_score < 0.01:
                suggestions.append(
                    f"Low mutual information with target (MI={mi_score:.2f}) — may not help predictive power."
                )

        insights[feature] = {
            "feature": feature,
            "description": dict_descriptors.get(feature, ""),
            "related_features": "<br>".join(related) if related else "None",
            "mutual_info with target": (
                f"{mi_score:.3f}" if mi_score is not None else "N/A"
            ),
            "suggestions": (
                "- " + "<br>- ".join(suggestions) if suggestions else "None"
            ),
        }

    return insights, num_features


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

    suggestions = ["Consider replacing with binary encoding (0/1)"]

    if col_data.nunique(dropna=True) == 1:
        suggestions.append("Drop constant boolean column")

    # Handle empty or NA case safely
    if len(col_non_null) > 0:
        true_ratio = col_non_null.mean()
        if pd.notna(true_ratio) and (true_ratio < 0.05 or true_ratio > 0.95):
            suggestions.append(
                "Class imbalance (may need resampling or weight)"
            )

    return suggestions


def safe_skew(values):
    values = pd.Series(values).dropna()
    if len(values) < 3:
        return np.nan
    if np.isclose(np.std(values), 0):
        return 0.0
    return skew(values)


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
        skewness_non_zero = safe_skew(non_zero_values)
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


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded HTML image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_base64}" class="responsive-img"/>'


def generate_eda_plots(
    df, column_name, inferred_type, target="", verbose=False
):
    MAX_ROWS = 5000
    TOP_CATEGORIES = 20

    # Sample for performance
    sample_frac = 1.0 if len(df) <= MAX_ROWS else MAX_ROWS / len(df)
    df_sampled = (
        df.sample(frac=sample_frac, random_state=42)
        if sample_frac < 1.0
        else df.copy()
    )

    if sample_frac < 1.0 and verbose:
        print(
            f"[INFO] Sampled {int(sample_frac * 100)}% of data for plotting..."
        )

    col = df_sampled[column_name].dropna()
    target_col = df_sampled.loc[col.index, target] if target else None

    plot1_html = plot2_html = ""

    # Determine target type
    if target:
        target_type = pd.api.types.infer_dtype(df_sampled[target], skipna=True)
        is_target_numeric = target_type in [
            "integer",
            "floating",
            "mixed-integer-float",
        ]
    else:
        is_target_numeric = False

    # Truncate high-cardinality categories
    if inferred_type in ["category", "boolean", "string"]:
        top_categories = col.value_counts().nlargest(TOP_CATEGORIES).index
        col = col[col.isin(top_categories)]
        df_sampled = df_sampled[df_sampled[column_name].isin(top_categories)]

    # Plot 1: Distribution / Frequency
    if inferred_type in ["category", "boolean", "string"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x=col, order=col.value_counts().index, ax=ax)
        ax.set_title(f"Frequency of {column_name}")
        ax.set_xlabel(column_name)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.patch.set_alpha(0.0)
        plot1_html = fig_to_base64(fig)

    elif inferred_type in ["integer", "float"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            df_sampled[column_name].dropna(), bins=60, kde=False, ax=ax
        )
        ax.set_title(f"Distribution of {column_name}")
        ax.set_xlabel(column_name)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        plot1_html = fig_to_base64(fig)

    # Plot 2: Relation to target
    if target and column_name != target:
        if inferred_type in ["category", "boolean", "string"]:
            if is_target_numeric:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=column_name, y=target, data=df_sampled, ax=ax)
                ax.set_title(f"{target} per {column_name}")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                plot2_html = fig_to_base64(fig)
            else:
                crosstab = (
                    df_sampled.groupby([column_name, target], observed=False)
                    .size()
                    .reset_index(name="count")
                )
                fig, ax = plt.subplots(figsize=(8, 4))
                crosstab_pivot = crosstab.pivot(
                    index=column_name, columns=target, values="count"
                ).fillna(0)
                crosstab_pivot.plot(kind="bar", stacked=True, ax=ax)
                ax.set_title(f"{target} distribution per {column_name}")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                plot2_html = fig_to_base64(fig)

        elif inferred_type in ["integer", "float"]:
            feature_col = df_sampled[column_name].dropna()
            target_col = df_sampled.loc[feature_col.index, target]

            if is_target_numeric:
                scatter_df = pd.DataFrame(
                    {column_name: feature_col, target: target_col}
                )
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.scatterplot(
                    data=scatter_df,
                    x=column_name,
                    y=target,
                    alpha=0.3,
                    s=10,
                    ax=ax,
                )
                ax.set_title(f"{target} vs {column_name}")
                fig.tight_layout()
                plot2_html = fig_to_base64(fig)
            else:
                try:
                    df_sampled["binned_feature"] = pd.qcut(
                        df_sampled[column_name], q=10, duplicates="drop"
                    )
                    grouped = (
                        df_sampled.groupby("binned_feature", observed=False)[
                            target
                        ]
                        .mean()
                        .reset_index()
                    )
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(
                        data=grouped, x="binned_feature", y=target, ax=ax
                    )
                    ax.set_title(f"Mean {target} by binned {column_name}")
                    ax.tick_params(axis="x", rotation=45)
                    fig.tight_layout()
                    plot2_html = fig_to_base64(fig)
                except Exception as e:
                    plot2_html = f"<p>Could not generate binned plot: {e}</p>"

    elif target == column_name:
        plot2_html = f"<p>No plot as feature is target ({target}).</p>"

    return {"plot1": plot1_html, "plot2": plot2_html}


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

    def analyze_parts(samples, depth=0):
        delimiter_regex = re.compile(r"[-_/:|]")
        if not samples:
            return
        indent = "&nbsp;&nbsp;" * (depth - 1)
        # Step 1: Find common delimiters across all samples
        delimiter_counts = {}
        for s in samples:
            for match in delimiter_regex.finditer(s):
                delimiter_counts[match.group()] = (
                    delimiter_counts.get(match.group(), 0) + 1
                )

        # Pick delimiter that is present in all samples
        common_delim = None
        for match in delimiter_regex.finditer(samples[0]):
            delim = match.group()
            if all(delim in s for s in samples):
                common_delim = delim
                break

        if not common_delim:
            if depth == 0:
                return
            # No consistent delimiter; check the remaining strings
            lengths = [len(s) for s in samples]
            all_numeric = all(s.isdigit() for s in samples)
            if all_numeric:
                if len(set(lengths)) == 1:
                    suggestions.append(
                        f"#{indent}- Right part is numeric of consistent length {lengths[0]}."
                    )
                else:
                    suggestions.append(
                        f"#{indent}- Right part is numeric (variable length allowed)."
                    )
            else:
                if len(set(lengths)) == 1:
                    suggestions.append(
                        f"#{indent}- Right part is non-numeric string of consistent length {lengths[0]}."
                    )
                else:
                    suggestions.append(
                        f"#{indent}- Right part is non-numeric string with varying lengths: {set(lengths)}."
                    )
            return

        # Split strings into 2 parts
        left_parts = []
        right_parts = []
        for s in samples:
            split_index = s.find(common_delim)
            left_parts.append(s[:split_index])
            right_parts.append(s[split_index + 1 :])
        if depth == 0:
            suggestions.append(f"Split on '{common_delim}'")
        else:
            suggestions.append(f"#{indent}- Split on '{common_delim}'")
        indent = "&nbsp;&nbsp;" * (depth)
        # Analyze left part
        all_numeric = all(part.isdigit() for part in left_parts)
        lengths = [len(p) for p in left_parts]
        if all_numeric:
            if len(set(lengths)) == 1:
                suggestions.append(
                    f"#{indent}- Left part is numeric of consistent length {lengths[0]}."
                )
            else:
                suggestions.append(
                    f"#{indent}- Left part is numeric (variable length allowed)."
                )
        else:

            if len(set(lengths)) == 1:
                suggestions.append(
                    f"#{indent}- Left part is non-numeric string of consistent length {lengths[0]}."
                )
            else:
                suggestions.append(
                    f"#{indent}- Left part is non-numeric string with varying lengths: {set(lengths)}."
                )

        # Recurse on right part
        analyze_parts(right_parts, depth=depth + 1)

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
    analyze_parts([str(s) for s in series])

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


def suggest_categorical_handling(entropy, cardinality):
    """
    Suggests preprocessing strategy based on entropy and cardinality.

    Parameters:
    - entropy (float): Entropy of the feature (0 = very predictable, high = unpredictable)
    - cardinality (int): Number of unique non-null values in the feature

    Returns:
    - str: A human-readable suggestion
    """

    # Classify cardinality
    if cardinality <= 5:
        cardinality_class = "low"
    elif cardinality <= 20:
        cardinality_class = "medium"
    else:
        cardinality_class = "high"

    # Classify entropy
    if entropy < 0.5:
        entropy_class = "low"
    elif entropy < 2.0:
        entropy_class = "medium"
    else:
        entropy_class = "high"

    # Suggestions matrix
    suggestions = {
        (
            "low",
            "low",
        ): "Very predictable and limited values. Consider binary encoding, or drop if dominated by a single value.",
        (
            "low",
            "medium",
        ): "A few common values with some variation. One-hot encoding or ordinal encoding may work well.",
        (
            "low",
            "high",
        ): "Rare case. Check for label issues or merge categories. One-hot may still be possible.",
        (
            "medium",
            "low",
        ): "Moderate number of values but low variation. Consider frequency encoding or grouping infrequent values.",
        ("medium", "medium"): "Good candidate for one-hot or ordinal encoding.",
        (
            "medium",
            "high",
        ): "High diversity and moderate value count. Frequency encoding or target encoding is advised.",
        (
            "high",
            "low",
        ): "Many categories but one dominant. Consider reducing to 'Other' + top categories or dropping.",
        (
            "high",
            "medium",
        ): "High cardinality and some variation. Use frequency encoding or reduce dimensionality first.",
        (
            "high",
            "high",
        ): "Very diverse feature. Avoid one-hot encoding. Use target, frequency, or hashing encoding.",
    }

    return suggestions.get(
        (cardinality_class, entropy_class), "No suggestion available."
    )


def analyze_categorical_column(df: pd.DataFrame, column_name: str) -> list[str]:
    suggestions = []

    s = df[column_name]
    n_unique = s.nunique(dropna=True)
    freq = s.value_counts(normalize=True, dropna=True)
    value_counts = df[column_name].value_counts(normalize=True, dropna=True)
    top_freq = freq.iloc[0] if not freq.empty else 0
    entropy_value = scipy_entropy(value_counts, base=2)  # base-2: bits
    suggestions = []
    suggestions.append(suggest_categorical_handling(entropy_value, n_unique))
    # Cardinality-based suggestions
    if n_unique == 1:
        suggestions.append(
            f"Column '{column_name}' is constant. Consider dropping it."
        )

    # Frequency distribution
    if top_freq > 0.9:
        suggestions.append(
            f"Column '{column_name}' is dominated by a single category ({top_freq:.1%} frequency). Consider grouping rare categories."
        )

    return suggestions
