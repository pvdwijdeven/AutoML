# Standard library imports
from typing import Literal, Union, Any, Optional

# Third-party imports
from pandas import Series, DataFrame
from scipy.stats import shapiro
import numpy as np
from automl.library import Logger


def handle_outliers(
    X: DataFrame,
    y: Optional[Series],
    fit: bool,
    step_params: dict[str, Any],
    before: bool,
    logger: Logger,
    **config: Any,
) -> tuple[DataFrame, Optional[Series], dict[str, Any]]:
    """
    Handles outliers for numeric columns in the DataFrame X.

    This function operates in two modes:
    - fit=True: Determines the strategy (capping or imputation), calculates
        the required bounds/imputation value, and stores them in step_params.
    - fit=False: Loads the stored parameters and applies the transformation.

    The transformation is applied to X  in both modes.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix.
    y : Optional[pd.Series]
        The target variable (not used for outlier calculation but passed through).
    fit : bool
        If True, the function is in the fitting/training phase.
    step_params : dict[str, Any]
        A dictionary to store/retrieve learned parameters (strategies, bounds, etc.).
    logger : Logger
        The logging utility object.
    meta_data : dict[str, dict[str, Any]]
        Metadata about the dataset and features (not directly used here but passed through).
    **config : Any
        Global configuration settings, which may contain default parameters.

    Returns
    -------
    tuple[pd.DataFrame, Optional[pd.Series], dict[str, Any]]
        The transformed feature matrix X, the original target y, and the updated step_params.
    """
    X_work = X.copy()

    if fit:
        for col in X_work.select_dtypes(include=np.number).columns:
            # Use the intelligent decision function
            if (step_params["outlier_order"][col] == "outliers_first") == before:
                outlier_params = decide_outlier_handling_method(column=X_work[col])
                step_params[f"outliers_{'before' if before else 'after'}"][col] = outlier_params

    for col, params in step_params[f"outliers_{'before' if before else 'after'}"].items():
        if col not in X_work.columns:
            continue

        series = X_work[col]
        original_dtype = series.dtype

        strategy = params[0]

        if strategy == "ignore":
            # No action needed
            continue

        elif strategy == "cap":
            # params = ("cap", lower_bound, upper_bound)
            _, lower_bound, upper_bound = params

            # Capping/Clipping transformation
            X_work[col] = series.clip(lower=lower_bound, upper=upper_bound)

        elif strategy == "impute":
            # params = ("impute", lower_bound, upper_bound, median_value)
            _, lower_bound, upper_bound, median_value = params

            # Identify outliers
            outliers_mask = (series < lower_bound) | (series > upper_bound)

            # Replace outliers with the pre-calculated median value
            X_work.loc[outliers_mask, col] = median_value

        # Preserve original dtype (e.g., if it was a pandas 'Int64' for nullable integers)
        X_work[col] = X_work[col].astype(original_dtype)

    return X_work, y, step_params


def decide_outlier_handling_method(
    column: Series,
    min_large_dataset: int = 1000,
    max_drop_outlier_pct: float = 0.05,  # kept for compatibility, now influences "cap" vs "impute"
    missing_threshold: float = 0.1,
    extreme_outlier_factor: float = 2.0,
) -> Union[
    Literal["ignore"],
    tuple[Literal["impute"], float, float, float],
    tuple[Literal["cap"], float, float],
]:
    """
    Decide the method to handle outliers in a numeric column without dropping rows.

    Since dropping rows is not allowed in a transform pipeline context, this function
    non-destructively chooses between capping or imputing outliers.

    Logic Summary
    -------------
    1. If not enough data (< 3 samples) → "ignore".
    2. If **extreme outliers** exist (outside extreme_outlier_factor * standard bounds)
       → **"cap"** using the *extreme* bounds (a more conservative clipping).
    3. If **missingness** is high (> missing_threshold) OR **outlier proportion**
       is moderate/high (> max_drop_outlier_pct)
       → **"impute"** by replacing outliers with the median of the non-outlier data.
    4. Otherwise (mild outliers, tolerable missingness)
       → **"cap"** using the *standard* bounds (e.g., 1.5*IQR or 3*std).

    Parameters
    ----------
    column : pd.Series
        Numeric column to evaluate.
    min_large_dataset : int, default=1000
        (Retained for compatibility, but not used in the decision logic).
    max_drop_outlier_pct : float, default=0.05
        Threshold for deciding whether outliers are "moderate/high"; used to switch
        between "cap" vs "impute".
    missing_threshold : float, default=0.1
        If missing ratio > threshold → prefer "impute".
    extreme_outlier_factor : float, default=2.0
        Multiplier to expand bounds for detecting (and sometimes capping) extreme outliers.

    Returns
    -------
    Union[Literal["ignore"],
          tuple[Literal["impute"], float, float, float],
          tuple[Literal["cap"], float, float]]
        Strategy and necessary parameters:
        - "ignore"
        - ("cap", lower_bound, upper_bound): Clips values below/above the bounds.
        - ("impute", lower_bound, upper_bound, median_value): Replaces values
          below/above the bounds with the `median_value` (the median of the normal data).

    Notes
    -----
    - Uses `get_threshold_method` (IQR or Z-score) for standard boundary calculation.
    - Dropping rows is explicitly disallowed in this context.
    """
    # Drop NaNs for outlier evaluation
    data = column.dropna()
    n = len(data)
    missing_pct = (len(column) - n) / len(column) if len(column) > 0 else 0

    # Not enough data → safest option
    if n < 3:
        return "ignore"

    # Outlier detection method
    # NOTE: Assumes get_threshold_method(data) is available
    threshold_method = get_threshold_method(data)

    # Compute outlier bounds
    if threshold_method == "iqr":
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        # Extreme bounds are a factor wider
        extreme_lower = Q1 - extreme_outlier_factor * 1.5 * IQR
        extreme_upper = Q3 + extreme_outlier_factor * 1.5 * IQR

    elif threshold_method == "zscore":
        mean, std = data.mean(), data.std(ddof=0)
        lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
        # Extreme bounds are a factor wider
        extreme_lower = mean - extreme_outlier_factor * 3 * std
        extreme_upper = mean + extreme_outlier_factor * 3 * std

    else:  # Fallback
        return "ignore"

    # Detect outliers
    outliers = (data < lower_bound) | (data > upper_bound)
    # Normal data includes the boundary points for robust median calculation
    normal = (data >= lower_bound) & (data <= upper_bound)
    outlier_pct = outliers.mean()
    extreme_outlier_count = (
        (data < extreme_lower) | (data > extreme_upper)
    ).sum()

    # 1. Extreme outliers → cap at the extreme bounds
    if extreme_outlier_count > 0:
        return "cap", extreme_lower, extreme_upper

    # 2. Significant missingness or too many outliers → impute
    # FIX: Calculate the median of the *normal data* points
    col_median = data[normal].median()
    if missing_pct > missing_threshold or outlier_pct > max_drop_outlier_pct:
        # Return standard bounds (1.5*IQR or 3*std) and the median of normal data
        return "impute", lower_bound, upper_bound, col_median

    # 3. Otherwise, mild outliers → cap safely at the standard bounds
    return "cap", lower_bound, upper_bound


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
