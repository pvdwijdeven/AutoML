# Standard library imports
from typing import Any, Literal, Optional

# Third-party imports
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import shapiro

# Local application imports
from automl.eda.dataset_overview import find_duplicate_columns
from automl.library import Logger
from .missing_values import decide_imputation_strategy


def general_info(
    X: DataFrame,
    y: Optional[Series],
    fit: bool,
    step_params: dict[str, Any],
    logger: Logger,
    meta_data: dict[str, dict[str, Any]],
    **config: Any,
) -> tuple[DataFrame, Optional[Series], dict[str, Any]]:
    if fit:
        step_params["missing_imputation"] = decide_imputation_strategy(
            X_train=X, logger=logger
        )
        step_params["outlier_order"] = {}
        for col in X.columns:
            step_params["outlier_order"][col] = decide_outlier_order(
                series=X[col],
                imputation_method=step_params["missing_imputation"][col].method,
            )
    return X, y, step_params


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


def drop_constant_columns(
    X: DataFrame,
    y: Optional[Series],
    fit: bool,
    step_params: dict[str, Any],
    logger: Logger,
    meta_data: dict[str, dict[str, Any]],
    **config: Any,
) -> tuple[DataFrame, Optional[Series], dict[str, Any]]:
    """
    Identifies and drops columns where all values are the same (constant/single-valued).

    For integration into the AutomlTransformer pipeline, it uses the fit/transform
    pattern to ensure consistency across training and test data.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature data.
    y : Optional[pd.Series]
        Input target data (not modified).
    fit : bool
        If True, identify columns to drop and store them in step_params.
        If False, drop columns specified in step_params.
    step_params : dict[str, Any]
        Dictionary to store/retrieve the list of constant columns to drop.
    logger : Logger
        Custom logger instance.
    meta_data : dict[str, dict[str, Any]]
        Current global metadata (unused in this specific step, but required by signature).
    **config : Any
        Additional configuration parameters (unused).

    Returns
    -------
    tuple[pd.DataFrame, Optional[pd.Series], dict[str, Any]]
        The transformed X, unmodified y, and updated step_params.
    """

    COLUMNS_TO_DROP_KEY = "constant_columns_to_drop"

    if fit:
        logger.info("Step: Identifying constant columns to drop.")

        # 1. Identify constant columns
        # Use nunique(dropna=True) to count unique values, excluding NaNs as a unique value
        # If nunique is 1, the column is constant (e.g., [1, 1, 1] or [NaN, NaN, NaN])
        # If nunique is 0, the column is entirely empty (should be caught by missing value logic later,
        # but 0-nunique columns are often only possible with old pandas versions or object dtypes)

        nunique_series = X.nunique(dropna=True)

        # A constant column has exactly one unique value.
        constant_cols = list(nunique_series[nunique_series <= 1].index)

        # 2. Store the list of columns to drop for the transform phase
        step_params[COLUMNS_TO_DROP_KEY] = constant_cols

        logger.debug(
            f"Found {len(constant_cols)} constant columns: {constant_cols}"
        )

    else:
        logger.info("Step: Applying constant column drop.")

    # --- Transformation (Applies in both fit=True and fit=False) ---

    # 3. Retrieve the list of columns to drop
    columns_to_drop = step_params.get(COLUMNS_TO_DROP_KEY, [])

    # Filter the list to only include columns currently present in X
    valid_cols_to_drop = [col for col in columns_to_drop if col in X.columns]

    if valid_cols_to_drop:
        X_out = X.drop(columns=valid_cols_to_drop, inplace=False)
        logger.info(
            f"Dropped {len(valid_cols_to_drop)} constant column(s): {valid_cols_to_drop}"
        )
    else:
        X_out = X.copy()
        if fit:
            logger.info("No constant columns found.")
        else:
            logger.info("No stored constant columns to drop.")

    return X_out, y, step_params


def drop_duplicate_columns(
    X: DataFrame,
    y: Optional[Series],
    fit: bool,
    step_params: dict[str, Any],
    logger: Logger,
    meta_data: dict[str, dict[str, Any]],
    **config: Any,
) -> tuple[DataFrame, Optional[Series], dict[str, Any]]:
    """
    Identifies and drops columns that are duplicates of another column.

    It preserves the first column in a set of duplicates and drops all subsequent ones.
    For integration into the AutomlTransformer pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature data.
    y : Optional[pd.Series]
        Input target data (not modified).
    fit : bool
        If True, identify columns to drop and store them in step_params.
        If False, drop columns specified in step_params.
    step_params : dict[str, Any]
        Dictionary to store/retrieve the list of columns to drop.
    logger : Logger
        Custom logger instance.
    meta_data : dict[str, dict[str, Any]]
        Current global metadata (unused in this specific step, but required by signature).
    **config : Any
        Additional configuration parameters (unused).

    Returns
    -------
    tuple[pd.DataFrame, Optional[pd.Series], dict[str, Any]]
        The transformed X, unmodified y, and updated step_params.
    """

    COLUMNS_TO_DROP_KEY = "duplicate_columns_to_drop"

    if fit:
        logger.info("Step: Identifying duplicate columns to drop.")

        # 1. Identify all duplicate relationships
        duplicates_dict = find_duplicate_columns(X_train=X)

        # 2. Extract the unique list of columns to be dropped (the 'other_col' in the dict values)
        # Note: By iterating through X.columns in the order they appear, the column that appears first
        # will always be kept as the 'master' key, and its duplicates (in the list) will be dropped.
        columns_to_drop = set()
        for col_list in duplicates_dict.values():
            columns_to_drop.update(col_list)

        columns_to_drop_list = sorted(
            list(columns_to_drop)
        )  # Store as a sorted list for consistency

        # 3. Store the list of columns to drop for the transform phase
        step_params[COLUMNS_TO_DROP_KEY] = columns_to_drop_list

        logger.debug(
            f"Found {len(columns_to_drop_list)} duplicate columns: {columns_to_drop_list}"
        )

    else:
        logger.info("Step: Applying duplicate column drop.")

    # --- Transformation (Applies in both fit=True and fit=False) ---

    # 4. Retrieve the list of columns to drop
    columns_to_drop = step_params.get(COLUMNS_TO_DROP_KEY, [])

    # Filter the list to only include columns currently present in X
    # This prevents errors if a previous step already dropped one of the columns (e.g., constant columns)
    valid_cols_to_drop = [col for col in columns_to_drop if col in X.columns]

    if valid_cols_to_drop:
        X_out = X.drop(columns=valid_cols_to_drop, inplace=False)
        logger.info(
            f"Dropped {len(valid_cols_to_drop)} duplicate column(s): {valid_cols_to_drop}"
        )
    else:
        X_out = X.copy()
        if fit:
            logger.info("No duplicate columns found.")
        else:
            logger.info("No stored duplicate columns to drop.")

    return X_out, y, step_params


def decide_outlier_order(
    series: Series, imputation_method: str
) -> Literal["impute_first", "outliers_first"]:
    """
    Decides the order of outlier handling based on a column's properties.

    This function implements the logic from the user-provided flowchart.

    Args:
        series: The column (pandas Series) to analyze.
        imputation_method: The planned imputation method ('mean', 'median', 'MICE', 'none').

    Returns:
        A string literal: 'impute_first' or 'outliers_first'.
    """
    # --- 1. Define Thresholds and Initial Checks ---
    if not pd.api.types.is_numeric_dtype(series):
        # Outlier handling is not applicable to non-numeric types
        return "impute_first"

    HIGH_MISSING_THRESHOLD = 0.20  # 20%
    DATA_ERROR_FACTOR = 10.0  # Heuristic for detecting data entry errors

    missing_percentage = series.isnull().sum() / len(series)
    non_null_series = series.dropna()

    # --- 2. Follow the Decision Tree Logic ---

    # B: Is the percentage of missing data high?
    if missing_percentage > HIGH_MISSING_THRESHOLD:
        # Path forks to 'Yes' -> C

        # C: Are the outliers likely data entry errors? (Heuristic)
        # We check if the max value is drastically larger than the 99th percentile.
        if len(non_null_series) < 10:
            # Not enough data for a reliable heuristic, default to safer option
            return "outliers_first"

        p99 = non_null_series.quantile(0.99)
        max_val = non_null_series.max()

        # Check for non-zero p99 to avoid division by zero
        if p99 > 0 and (max_val / p99) > DATA_ERROR_FACTOR:
            # Path forks to 'Yes' -> E
            return "outliers_first"  # Treat likely data errors first
        else:
            # Path forks to 'No' -> F
            return "outliers_first"  # Default for high missingness without clear errors
    else:
        # Path forks to 'No' -> D

        # D: Are you using a robust imputation method?
        robust_methods = {"median", "mode", "MICE"}
        if imputation_method in robust_methods:
            # Path forks to 'Yes' -> G
            return "impute_first"
        else:
            # Path forks to 'No' -> H (e.g., for 'mean' imputation)
            return "outliers_first"
