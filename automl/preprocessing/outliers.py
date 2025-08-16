from library import Logger

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Literal, List
from scipy.stats import shapiro


def get_threshold_method(
    column: pd.Series, max_sample_size: int = 5000
) -> Literal["iqr"] | Literal["zscore"]:
    """
    Determine the appropriate statistical method to detect outliers for a specific
    column. Uses the Shapiro-Wilk test for normality on sample sizes within
    the `max_sample_size` limit.

    Args:
        column (pd.Series): Data column for normality test.
        max_sample_size (int): Maximum size to run Shapiro-Wilk test.

    Returns:
        str: 'zscore' if data is approximately normal and sample size is manageable;
            'iqr' otherwise.
    """
    # Drop NaN values before normality test
    data = column.dropna()

    # Use IQR if too few data points or too many
    if len(data) < 3 or len(data) > max_sample_size:
        return "iqr"  # Skip Shapiro for large or too small datasets

    stat, p_value = shapiro(data)

    if p_value > 0.05:
        return "zscore"
    else:
        return "iqr"


def outlier_imputation_order(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit: bool,
    step_params: Dict[str, Any],
    target_aware: bool = False,
    logger: Logger,
    step_outputs: Dict[str, Any],
    missing_threshold: float = 0.1,
    extreme_outlier_factor: float = 2.0,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Decide whether outliers in a specific column should be handled before or after
    missing value imputation.

    The decision is based on:
        - Presence of extreme outliers.
        - Proportion of outliers relative to missing data.
        - Distribution shape estimated by IQR or Z-score methods.

    Args:
        column (pd.Series): Column data including missing values.
        missing_threshold (float): Missing value proportion threshold to consider low/high.
        extreme_outlier_factor (float): Factor to enlarge thresholds for extreme outliers.

    Returns:
        str: 'before_imputation' or 'after_imputation' indicating outlier handling timing.
    """
    if fit or not step_outputs["skip_outliers"]["skip_outliers"]:
        before_or_after = {}
        for columnname in X.columns:
            column = X[columnname].copy()
            if not pd.api.types.is_numeric_dtype(column):
                before_or_after[columnname] = "no_encoding"
                continue
            data = column.dropna()
            total_count = len(column)
            missing_pct = (
                (total_count - len(data)) / total_count
                if total_count > 0
                else 0
            )

            if len(data) < 3:
                # Not enough data to decide, default to after imputation
                before_or_after[columnname] = "after_imputation"
                continue

            threshold_method = get_threshold_method(data)

            # Calculate bounds for outlier detection, according to method
            if threshold_method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                # Standard bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Amplified bounds to detect extreme outliers
                extreme_lower = Q1 - extreme_outlier_factor * 1.5 * IQR
                extreme_upper = Q3 + extreme_outlier_factor * 1.5 * IQR

            elif threshold_method == "zscore":
                mean = data.mean()
                std = data.std()
                # Standard bounds
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                # Amplified bounds for extreme outliers
                extreme_lower = mean - extreme_outlier_factor * 3 * std
                extreme_upper = mean + extreme_outlier_factor * 3 * std

            else:
                # Unsupported method fallback
                before_or_after[columnname] = "after_imputation"
                continue

            # Detect outliers using the standard bounds:
            outliers = (data < lower_bound) | (data > upper_bound)
            outliers_count = outliers.sum()
            outliers_pct = outliers_count / len(data)

            # Detect extreme outliers using amplified bounds:
            extreme_outliers = (data < extreme_lower) | (data > extreme_upper)
            extreme_outliers_count = extreme_outliers.sum()

            # Decision logic:

            # Handle outliers BEFORE imputation if:
            # - The column has extreme outliers (significant number)
            # - OR the proportion of outliers is high and missingness is low (outliers probably errors)
            if extreme_outliers_count > 0:
                before_or_after[columnname] = "before_imputation"
                continue

            if outliers_pct > 0.1 and missing_pct < missing_threshold:
                # More than 10% outliers and missingness low means outliers likely distort
                before_or_after[columnname] = "before_imputation"
                continue

                # Otherwise, handle outliers after imputation
            before_or_after[columnname] = "after_imputation"

        return (
            X,
            y,
            {
                "before_or_after": before_or_after,
                "skip_outliers": step_outputs["skip_outliers"]["skip_outliers"],
            },
        )
    else:
        return X, y, step_params


def decide_outlier_handling_method(
    column: pd.Series,
    min_large_dataset: int = 1000,
    max_drop_outlier_pct: float = 0.05,
    missing_threshold: float = 0.1,
    extreme_outlier_factor: float = 2.0,
) -> Literal["impute"] | Literal["cap"] | Literal["drop"]:
    """
    Decide how to treat outliers in a given column: drop rows, cap values, or impute.

    Logic depends on:
        - Dataset size.
        - Percent of outlier values.
        - Missing data proportion.
        - Presence of extreme outliers.

    Args:
        column (pd.Series): Column data with potential outliers.
        min_large_dataset (int): Dataset size for "large" classification.
        max_drop_outlier_pct (float): Max % outliers allowed for safe row dropping.
        missing_threshold (float): Threshold for high missingness to favor imputation.
        extreme_outlier_factor (float): Multiplier for extreme outlier detection.

    Returns:
        str: One of 'drop', 'cap', or 'impute' indicating chosen outlier handling method.
    """
    data = column.dropna()
    n = len(data)
    missing_pct = (len(column) - n) / len(column) if len(column) > 0 else 0

    if n < 3:
        # Not enough data to decide meaningfully, default to 'impute'
        return "impute"

    threshold_method = get_threshold_method(data)

    # Calculate bounds for outliers
    if threshold_method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        extreme_lower = Q1 - extreme_outlier_factor * 1.5 * IQR
        extreme_upper = Q3 + extreme_outlier_factor * 1.5 * IQR

    elif threshold_method == "zscore":
        mean = data.mean()
        std = data.std()

        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        extreme_lower = mean - extreme_outlier_factor * 3 * std
        extreme_upper = mean + extreme_outlier_factor * 3 * std

    else:
        # Fallback, safest to cap
        return "cap"

    outliers = (data < lower_bound) | (data > upper_bound)
    outlier_pct = outliers.sum() / n if n > 0 else 0

    extreme_outliers = (data < extreme_lower) | (data > extreme_upper)
    extreme_outlier_count = extreme_outliers.sum()

    # Decision logic:

    # 1. If dataset is large (>= min_large_dataset),
    #    and outlier percentage is small enough (<= max_drop_outlier_pct),
    #    and there are some outliers → drop rows to clean data aggressively.
    if (
        (n >= min_large_dataset)
        and (outlier_pct <= max_drop_outlier_pct)
        and (outlier_pct > 0)
    ):
        return "drop"

    # 2. If there are extreme outliers, but dropping too many rows would lose data, cap them.
    if extreme_outlier_count > 0:
        return "cap"

    # 3. If missing data proportion is high or outlier % is moderate/large,
    #    imputing is safer to avoid losing data or bias.
    if missing_pct > missing_threshold or outlier_pct > max_drop_outlier_pct:
        return "impute"

    # 4. Default fallback to cap
    return "cap"


def handle_outliers(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit: bool,
    step_params: Dict[str, Any],
    target_aware: bool = False,
    logger: Logger,
    step_outputs: Dict[str, Any],
    before: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Detects and handles outliers on specified columns of the training dataset.
    Applies same transformations consistently on validation, test, and optional test datasets.

    Supported methods: 'drop' (drop rows), 'cap' (clip values), 'impute' (median imputation).
    Supports threshold methods: 'iqr' or 'zscore'.

    Args:
        columns (List[str]): Columns to handle outliers on.
        before_or_after (str): Description of whether handling is before or after imputation.
        method (str): Optional explicit method for handling outliers; if empty auto-decides.
        threshold_method (str): Optional choice of threshold method; if empty auto-decides.
    """
    if fit:

        logger.debug(
            msg=f"[GREEN]- Handling outliers {'before missing values' if before else 'after missing values'} imputation"
        )
        X_train = X.copy()
        current_cols: List[str] = []
        to_find: str = "before_imputation" if before else "after_imputation"
        before_or_after = step_outputs["outlier_imputation_order"][
            "before_or_after"
        ]
        for col, value in before_or_after.items():
            if value == to_find:
                # remove columns that are already solved by encoding
                if col in X_train.columns:
                    current_cols.append(col)
        outlier_columns: Dict[str, Any] = {}

        for col in X_train[current_cols].columns:
            cur_threshold_method = get_threshold_method(column=X_train[col])
            if cur_threshold_method == "iqr":
                Q1 = X_train[col].quantile(0.25)
                Q3 = X_train[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            else:
                mean = X_train[col].mean()
                std = X_train[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std

            # Identify outliers in train set
            outliers_train = (X_train[col] < lower_bound) | (
                X_train[col] > upper_bound
            )

            cur_method = decide_outlier_handling_method(X_train[col])

            # if outliers_train.sum() == 0:
            #     outlier_columns[col] = {"method":"skip"}
            #     continue
            if cur_method == "drop":
                # Drop rows with outliers in train set
                X_train: pd.DataFrame = X_train.loc[~outliers_train]

                # Also drop these rows in y_train if applicable
                y = y.loc[X_train.index]
                outlier_columns[col] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "method": "drop",
                }
                logger.debug(
                    msg=f"[GREEN]- {outliers_train.sum()} outliers in column {col} dropped"
                )
            elif cur_method == "cap":
                X_train[col] = np.where(
                    X_train[col] < lower_bound, lower_bound, X_train[col]
                )
                X_train[col] = np.where(
                    X_train[col] > upper_bound, upper_bound, X_train[col]
                )
                outlier_columns[col] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "method": "cap",
                }
                logger.debug(
                    msg=f"[GREEN]  - {outliers_train.sum()} outliers in column {col} capped between {X_train[col].min()} and {X_train[col].max()}"
                )
            else:  # cur_method == "impute"
                median: float = X_train[col].median()

                X_train.loc[outliers_train, col] = median
                outlier_columns[col] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "method": "median",
                    "median": median,
                }
                logger.debug(
                    msg=f"[GREEN]  - {outliers_train.sum()} outliers in column {col} imputed with median {median}"
                )
        return X_train, y, {"outlier_columns": outlier_columns}
    else:
        for col, value in step_params["outlier_columns"].items():
            lower_bound = value["lower_bound"]
            upper_bound = value["upper_bound"]
            outliers_train = (X[col] < lower_bound) | (X[col] > upper_bound)
            if value["method"] == "drop":
                pass
            elif value["method"] == "cap":
                X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
                X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
            else:  # value["method"] == "median"
                X.loc[outliers_train, col] = value["median"]

        return X, y, {}
