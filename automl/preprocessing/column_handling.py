# Standard library imports
from typing import Any, Literal, Optional

# Third-party imports
import pandas as pd
from pandas import DataFrame, Series

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


def drop_strings(
    X: DataFrame,
    y: Optional[Series],
    fit: bool,
    step_params: dict[str, Any],
    logger: Logger,
    **config: Any,
) -> tuple[DataFrame, Optional[Series], dict[str, Any]]:
    """
    Drops string-like columns (string, category, object) with too many unique values.

    During the *fit* stage, the function identifies string-like columns whose number
    of unique values exceeds a threshold (`max_unique`) and records them in `step_params`.
    During the *transform* stage, the previously recorded columns are dropped.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : Optional[pd.Series]
        Target values, used to determine threshold for unique values.
    fit : bool
        Whether to operate in 'fit' mode (identify columns) or 'transform' mode (apply drops).
    step_params : dict[str, Any]
        dictionary for saving or retrieving parameters across fit/transform stages.
    logger : Logger
        Logger instance for debug messages.
    meta_data : dict[str, Any]
        Additional metadata dictionary (currently unused).

    Returns
    -------
    tuple[pd.DataFrame, Optional[pd.Series], Optional[dict[str, Any]]]
        Transformed (or unmodified) X, unchanged y, and updated step_params.
    """
    if fit:
        string_columns = []
        max_unique = min(
            20, max(10, int(0.01 * len(y)) if y is not None else 10)
        )
        for col in X.columns:
            if X[col].dtype in ["string", "category", "object"]:
                if X[col].nunique(dropna=True) > max_unique:
                    string_columns.append(col)
        step_params["drop_cols"] = string_columns

    string_columns = step_params.get("drop_cols", [])
    if string_columns:
        logger.debug(f"[GREEN]- Dropping string-like columns: {string_columns}")
        X = X.drop(columns=string_columns)

    return X, y, step_params


def drop_constant_columns(
    X: DataFrame,
    y: Optional[Series],
    fit: bool,
    step_params: dict[str, Any],
    logger: Logger,
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
