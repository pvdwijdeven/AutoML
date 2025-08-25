# Standard library imports
from typing import Any, Dict, Literal, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from scipy.stats import shapiro

# Local application imports
from automl.library import Logger


def get_threshold_method(
    column: pd.Series,
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


def outlier_imputation_order(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    *,
    fit: bool,
    step_params: Dict[str, Any],
    logger: Logger,
    meta_data: Dict[str, Any],
    missing_threshold: float = 0.1,
    extreme_outlier_factor: float = 2.0,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Decide whether outliers in each numeric column should be handled before
    or after missing value imputation.

    Logic:
    - Non-numeric columns are ignored (labeled as "no_encoding").
    - Columns with insufficient data (< 3 samples) default to "after_imputation".
    - For numeric columns with enough data:
        - The outlier detection method is chosen by `get_threshold_method`:
          - "iqr": Interquartile Range method
          - "zscore": Standard normal distribution check
        - Standard outlier bounds are used to estimate outlier proportion.
        - Amplified bounds (by `extreme_outlier_factor`) detect extreme outliers.

    Decision criteria:
    - Handle outliers **before imputation** if:
        - Any extreme outliers are present, OR
        - At least 10% of values are outliers while missingness is low (< missing_threshold).
    - Otherwise, handle **after imputation**.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : Optional[pd.Series]
        Target vector (unchanged).
    fit : bool
        Whether called in "fit" mode (compute new logic) or "transform" mode (reuse parameters).
    step_params : Dict[str, Any]
        Dictionary for saving or reusing parameters across fit/transform stages.
    logger : Logger
        Logger for debug messages.
    meta_data : Dict[str, Any]
        Metadata dictionary. Must include "skip_outliers" key with {"skip_outliers": bool}.
    missing_threshold : float, default=0.1
        Maximum fraction of missing values tolerated when deciding "before imputation".
    extreme_outlier_factor : float, default=2.0
        Amplification factor for bounds when detecting extreme outliers.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]
        (Potentially unchanged) X, unchanged y, and updated step_params which includes:
        - "before_or_after": dict mapping column → {"before_imputation" | "after_imputation" | "no_encoding"}
        - "skip_outliers": bool flag from meta_data

    Notes
    -----
    - If `fit` is False and `skip_outliers` is True, the function simply returns the inputs unchanged.
    - This function depends on `get_threshold_method` for choosing outlier detection strategy.
    """
    skip_outliers = meta_data.get("skip_outliers", {}).get(
        "skip_outliers", False
    )

    if fit or not skip_outliers:
        before_or_after: Dict[str, str] = {}

        for column_name in X.columns:
            column = X[column_name].copy()

            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(column):
                before_or_after[column_name] = "no_encoding"
                continue

            # Drop NaN for analysis
            data = column.dropna()
            total_count = len(column)
            missing_pct = (
                (total_count - len(data)) / total_count
                if total_count > 0
                else 0
            )

            # Not enough data to decide -> defer outlier handling
            if len(data) < 3:
                before_or_after[column_name] = "after_imputation"
                continue

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
                before_or_after[column_name] = "after_imputation"
                continue

            # Detect standard and extreme outliers
            outliers = (data < lower_bound) | (data > upper_bound)
            outliers_pct = outliers.sum() / len(data)

            extreme_outliers = (data < extreme_lower) | (data > extreme_upper)
            extreme_outliers_count = extreme_outliers.sum()

            # Decision logic
            if extreme_outliers_count > 0:
                before_or_after[column_name] = "before_imputation"
            elif outliers_pct > 0.1 and missing_pct < missing_threshold:
                before_or_after[column_name] = "before_imputation"
            else:
                before_or_after[column_name] = "after_imputation"

        step_params = {
            "before_or_after": before_or_after,
            "skip_outliers": skip_outliers,
        }

        logger.debug(
            f"[GREEN]- Outlier imputation strategy decided for {len(before_or_after)} columns."
        )
        return X, y, step_params

    # If skipping outlier handling entirely
    logger.debug("[GREEN]- Skipping outlier handling based on meta_data.")
    return X, y, step_params


def decide_outlier_handling_method(
    column: pd.Series,
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


def handle_outliers(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    *,
    fit: bool,
    step_params: Dict[str, Any],
    logger: Logger,
    meta_data: Dict[str, Any],
    before: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Detect and handle outliers in numeric columns by either capping
    them within calculated bounds or imputing their values (typically median).
    This function does not drop any rows, ensuring feature-target alignment.

    Logic (fit/transform pattern)
    -----------------------------
    - During 'fit':
        1. Decide which columns require outlier handling "before_imputation" or "after_imputation"
           using meta_data['outlier_imputation_order']["before_or_after"].
        2. For relevant columns, detect outliers by chosen threshold method (IQr or Z-score).
        3. Decide for each column whether to cap or impute (median), as per decide_outlier_handling_method.
        4. Store the chosen bounds/method per column in step_params.

    - During 'transform':
        1. Use stored bounds/methods to cap or impute outliers in each column.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe to process.
    y : Optional[pd.Series]
        Target series (unchanged).
    fit : bool
        If True, method decides thresholds and stores processing instructions.
        If False, applies stored instructions.
    step_params : Dict[str, Any]
        Dictionary for storing/retrieving outlier handling details.
    logger : Logger
        Logger for status/debug output.
    meta_data : Dict[str, Any]
        Metadata dictionary from preprocessing pipeline, must contain outlier handling order.
    before : bool, default=True
        If True, process columns flagged for 'before_imputation';
        otherwise, process those flagged for 'after_imputation'.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]
        Updated X, unchanged y, and new step_params with per-column outlier handling info (on 'fit').
    """
    if fit:
        X_work = X.copy()
        # Determine which columns to process based on outlier imputation order
        requested_order = "before_imputation" if before else "after_imputation"
        before_or_after = meta_data.get("outlier_imputation_order", {}).get(
            "before_or_after", {}
        )
        # Only select columns in X (ignore those dropped earlier)
        columns_to_process = [
            col
            for col, order in before_or_after.items()
            if order == requested_order and col in X_work.columns
        ]

        outlier_columns: Dict[str, Any] = {}
        for col in columns_to_process:
            threshold_method = get_threshold_method(X_work[col])
            # Calculate bounds
            if threshold_method == "iqr":
                Q1, Q3 = X_work[col].quantile(0.25), X_work[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            else:  # zscore
                mean, std = X_work[col].mean(), X_work[col].std(ddof=0)
                lower_bound, upper_bound = mean - 3 * std, mean + 3 * std

            outlier_mask = (X_work[col] < lower_bound) | (
                X_work[col] > upper_bound
            )
            handling_method = decide_outlier_handling_method(X_work[col])

            if handling_method == "cap":
                X_work[col] = np.where(
                    X_work[col] < lower_bound, lower_bound, X_work[col]
                )
                X_work[col] = np.where(
                    X_work[col] > upper_bound, upper_bound, X_work[col]
                )
                outlier_columns[col] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "method": "cap",
                }
                logger.debug(
                    f"[GREEN] - {outlier_mask.sum()} outliers in column '{col}' capped "
                    f"between {lower_bound:.5g} and {upper_bound:.5g}"
                )
            else:  # "impute"
                median_val: float = X_work[col].median()
                X_work.loc[outlier_mask, col] = median_val
                outlier_columns[col] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "method": "median",
                    "median": median_val,
                }
                logger.debug(
                    f"[GREEN] - {outlier_mask.sum()} outliers in column '{col}' imputed with median {median_val:.5g}"
                )

        return X_work, y, {"outlier_columns": outlier_columns}
    else:
        logger.debug(
            f"[GREEN]- Handling outliers {'before missing values' if before else 'after missing values'} imputation"
        )
        for col, meta in step_params.get("outlier_columns", {}).items():
            lower_bound = meta["lower_bound"]
            upper_bound = meta["upper_bound"]
            outlier_mask = (X[col] < lower_bound) | (X[col] > upper_bound)
            if meta["method"] == "cap":
                X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
                X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
                logger.debug(
                    f"[GREEN] - {outlier_mask.sum()} outliers in column '{col}' capped between {lower_bound:.5g} and {upper_bound:.5g}"
                )
            else:  # "median" impute
                X.loc[outlier_mask, col] = meta["median"]
                logger.debug(
                    f"[GREEN] - {outlier_mask.sum()} outliers in column '{col}' imputed with median {meta['median']:.5g}"
                )
        return X, y, {}
