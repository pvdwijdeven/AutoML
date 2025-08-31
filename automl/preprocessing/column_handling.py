# Standard library imports
from typing import Literal

# Third-party imports
from pandas import Series
from pandas.api.types import is_numeric_dtype
from scipy.stats import shapiro


def decide_outlier_handling_method(
    column: Series,
    min_large_dataset: int = 1000,
    max_drop_outlier_pct: float = 0.05,  # kept for compatibility, now influences "cap" vs "impute"
    missing_threshold: float = 0.1,
    extreme_outlier_factor: float = 2.0,
) -> Literal["impute", "cap"]:
    """
    Decide the method to handle outliers in a numeric column without dropping rows.

    Since dropping rows is not allowed in a transform pipeline context (to avoid
    misalignment with the target variable), this function only returns either:
    - "cap": replace extreme values with threshold limits
    - "impute": replace detected outliers with imputed values

    Logic
    -----
    1. If not enough data (< 3 samples) → "impute"
    2. If extreme outliers exist → "cap"
    3. If missingness is high (> missing_threshold)
        OR outlier proportion is moderate/high (> max_drop_outlier_pct) → "impute"
    4. Otherwise (mild outliers, tolerable missingness) → "cap"

    Parameters
    ----------
    column : pd.Series
        Numeric column to evaluate.
    min_large_dataset : int, default=1000
        (Retained for compatibility, but no longer used to trigger "drop").
    max_drop_outlier_pct : float, default=0.05
        Threshold for deciding whether outliers are "moderate/high"; used to switch
        between "cap" vs "impute".
    missing_threshold : float, default=0.1
        If missing ratio > threshold → prefer "impute".
    extreme_outlier_factor : float, default=2.0
        Multiplier to expand bounds and detect extreme outliers.

    Returns
    -------
    Literal["impute", "cap"]
        Strategy to handle outliers:
        - "cap"    : clip/cap values within threshold bounds
        - "impute" : replace with imputed estimates

    Notes
    -----
    - Uses `get_threshold_method` (IQR or Z-score) for boundary calculation.
    - Dropping rows is explicitly disallowed in this context.
    """
    # Drop NaNs for outlier evaluation
    data = column.dropna()
    n = len(data)
    missing_pct = (len(column) - n) / len(column) if len(column) > 0 else 0

    # Not enough data → safest option
    if n < 3:
        return "impute"

    # Outlier detection method
    threshold_method = get_threshold_method(data)

    # Compute outlier bounds
    if threshold_method == "iqr":
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        extreme_lower = Q1 - extreme_outlier_factor * 1.5 * IQR
        extreme_upper = Q3 + extreme_outlier_factor * 1.5 * IQR

    elif threshold_method == "zscore":
        mean, std = data.mean(), data.std(ddof=0)
        lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
        extreme_lower = mean - extreme_outlier_factor * 3 * std
        extreme_upper = mean + extreme_outlier_factor * 3 * std

    else:  # Fallback
        return "cap"

    # Detect outliers
    outliers = (data < lower_bound) | (data > upper_bound)
    outlier_pct = outliers.mean()
    extreme_outlier_count = (
        (data < extreme_lower) | (data > extreme_upper)
    ).sum()

    # 1. Extreme outliers → cap
    if extreme_outlier_count > 0:
        return "cap"

    # 2. Significant missingness or too many outliers → impute
    if missing_pct > missing_threshold or outlier_pct > max_drop_outlier_pct:
        return "impute"

    # 3. Otherwise, mild outliers → cap safely
    return "cap"


def get_threshold_method(
    column: Series,
    max_sample_size: int = 5000,
) -> Literal["iqr", "zscore"]:
    """
    Determine the method for outlier detection based on the distribution of a column.

    This function applies a normality check to decide whether to use
    the Interquartile Range (IQR) method or the Z-score method for outlier detection.

    Logic:
    - If the column has fewer than 3 samples (too small) or more than `max_sample_size` samples (too large),
        default to IQR because Shapiro–Wilk normality test is unreliable at very small sizes
        and computationally expensive for very large datasets.
    - Otherwise, perform the Shapiro–Wilk test:
        - If p-value > 0.05 → treat as approximately normal → return "zscore"
        - Otherwise → return "iqr"

    Parameters
    ----------
    column : pd.Series
        Column of numerical values.
    max_sample_size : int, default=5000
        Maximum number of samples allowed for running the Shapiro–Wilk test.
        Larger datasets will default to "iqr".

    Returns
    -------
    Literal["iqr", "zscore"]
        Chosen method for outlier detection:
        - "zscore" → dataset is approximately normal
        - "iqr"    → dataset is non-normal or data size not suitable for Shapiro test
    """
    # Remove NaN values before testing
    data = column.dropna()

    # Guard against too small or too large datasets
    if len(data) < 3 or len(data) > max_sample_size:
        return "iqr"

    # Shapiro-Wilk normality test
    _, p_value = shapiro(data)

    return "zscore" if p_value > 0.05 else "iqr"


def outlier_implementation_order(
    column,
    missing_threshold: float = 0.1,
    extreme_outlier_factor: float = 2.0,
) -> Literal[
    "no outlier handling",
    "after missing imputation",
    "before missing imputation",
]:
    """
    Determine the appropriate stage to handle outliers in a numeric column.

    This function analyzes a pandas Series column to decide when outlier handling
    should occur relative to missing value imputation. It evaluates data characteristics,
    missing value percentage, and outlier presence and severity using either IQR or Z-score methods.

    Args:
        column (pd.Series): The data column to analyze for outliers.
        missing_threshold (float, optional): Maximum allowed proportion of missing values
            to consider handling outliers before imputation. Defaults to 0.1.
        extreme_outlier_factor (float, optional): Factor to adjust the threshold for extreme outliers.
            Defaults to 2.0.

    Returns:
        Literal["no outlier handling", "after missing imputation", "before missing imputation"]:
            A string indicating when to apply outlier treatment:
                - "no outlier handling": for non-numeric columns.
                - "after missing imputation": if data is insufficient or no extreme outliers.
                - "before missing imputation": if extreme outliers exist or significant outliers present and missing threshold allows.

    """
    column_work = column.copy()

    # Skip non-numeric columns
    if not is_numeric_dtype(arr_or_dtype=column_work):
        return "no outlier handling"

    # Drop NaN for analysis
    data = column_work.dropna()
    total_count = len(column_work)
    missing_pct = (
        (total_count - len(data)) / total_count if total_count > 0 else 0
    )

    # Not enough data to decide -> defer outlier handling
    if len(data) < 3:
        return "after missing imputation"

    # Pick thresholding method (IQR or Z-score)
    threshold_method = get_threshold_method(data)

    # Compute bounds using chosen method
    if threshold_method == "iqr":
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        extreme_lower = Q1 - extreme_outlier_factor * 1.5 * IQR
        extreme_upper = Q3 + extreme_outlier_factor * 1.5 * IQR

    elif threshold_method == "zscore":
        mean, std = data.mean(), data.std(ddof=0)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        extreme_lower = mean - extreme_outlier_factor * 3 * std
        extreme_upper = mean + extreme_outlier_factor * 3 * std

    else:  # Fallback
        return "after missing imputation"

    # Detect standard and extreme outliers
    outliers = (data < lower_bound) | (data > upper_bound)
    outliers_pct = outliers.sum() / len(data)

    extreme_outliers = (data < extreme_lower) | (data > extreme_upper)
    extreme_outliers_count = extreme_outliers.sum()

    # Decision logic
    if extreme_outliers_count > 0:
        return "before missing imputation"
    elif outliers_pct > 0.1 and missing_pct < missing_threshold:
        return "before missing imputation"
    else:
        return "after missing imputation"
