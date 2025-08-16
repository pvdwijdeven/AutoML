from library import Logger

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any


def detect_dataset_type(target: pd.DataFrame | pd.Series | np.ndarray) -> str:
    """
    Detect dataset type based on the target variable.

    Parameters:
    - target: array-like (1D or 2D), target values for supervised learning.

    Returns:
    - str: dataset type among:
    'binary_classification',
    'multi_class_classification',
    'multi_label_classification',
    'ordinal_regression',
    'regression'.
    """

    # Convert to pandas object for convenience
    if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
        target_df = target
    else:
        # If numpy array or list
        if target.ndim == 1:
            target_df = pd.Series(data=target)
        elif target.ndim == 2:
            target_df = pd.DataFrame(data=target)
        else:
            raise ValueError("Target must be 1D or 2D array-like")

    # Determine if multi-label (2D target)
    if isinstance(target_df, pd.DataFrame) and target_df.shape[1] > 1:
        # Check if binary indicators (0/1) in multiple columns -> multi-label classification
        unique_vals = pd.unique(values=target_df.values.ravel())
        if set(unique_vals).issubset({0, 1}):
            return "multi_label_classification"
        else:
            # Otherwise it's more complicated but treat as multi-label
            return "multi_label_classification"

    # For 1D target (Series)
    target_series = (
        target_df if isinstance(target_df, pd.Series) else target_df.iloc[:, 0]
    )

    # Check if target is numeric
    is_numeric: bool = pd.api.types.is_numeric_dtype(arr_or_dtype=target_series)

    # Get unique values
    unique_vals = target_series.dropna().unique()
    n_unique: int = len(unique_vals)

    # Heuristic: if numeric with many unique values -> regression
    if is_numeric and n_unique > 20:
        return "regression"

    # Check if target is categorical (string or int categories)
    # If only two unique classes
    if n_unique == 2:
        # Calculate the class distribution ratio
        counts: pd.Series[float] = target_series.value_counts(normalize=True)

        # Define threshold for "high imbalance" (e.g. minority class <= 5%)
        imbalance_threshold = 0.05

        # Check minority class proportion
        minority_class_ratio: float = counts.min()

        if minority_class_ratio <= imbalance_threshold:
            return "imbalanced_binary_classification"
        else:
            return "binary_classification"

    # More than two unique classes:
    # If target is categorical or int but with ordered discrete small values,
    # attempt to detect ordinal by checking if sorted unique_vals are numeric and contiguous
    if not is_numeric:
        # Non-numeric classes likely multi-class
        return "multi_class_classification"
    else:
        # Numeric case:
        unique_vals_sorted = np.sort(a=unique_vals)
        diffs = np.diff(a=unique_vals_sorted)
        # Check if differences are all 1 and values are integers => ordinal (e.g. ratings)
        if np.all(a=diffs == 1) and np.all(
            a=unique_vals_sorted == unique_vals_sorted.astype(dtype=int)
        ):
            # Assume ordinal regression if unique count between 3 and 20
            if 3 <= n_unique <= 20:
                return "ordinal_regression"
            else:
                return "multi_class_classification"
        else:
            # Otherwise treat as multi-class classification if discrete numeric classes
            if n_unique <= 20:
                return "multi_class_classification"
            else:
                # If numeric with many unique classes but less than 20 (caught above regression >20)
                return "multi_class_classification"

    # Fallback (should not get here)
    return "unknown"


def skip_outliers(
    target: pd.Series,
) -> Dict[str, bool]:
    """
    Decide whether to skip outlier handling based on target distribution.

    This heuristic checks for imbalanced classification problems by
    examining the number of unique classes in the target and the frequency
    of the rarest classes.

    Returns:
        bool: True if outlier handling should be skipped due to imbalanced classes,
            False otherwise.
    """
    unique_classes = target.unique()
    total_samples = len(target)
    if len(unique_classes) <= 5:
        for cls in unique_classes:
            freq = (target == cls).sum() / total_samples
            if freq < 0.01:
                return {"skip_outliers": True}
    return {"skip_outliers": False}


def drop_duplicate_rows(
    X: pd.DataFrame,
    y: pd.Series,
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """
    Remove duplicate rows from the feature matrix `X` and target vector `y` where both match.

    This function identifies duplicate samples based on the combination of `X` and `y`.
    The first occurrence of each unique sample (features and target) is kept, while all
    subsequent duplicates are removed to maintain dataset consistency. Information about
    the number of dropped rows and their indices is logged and returned.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing the independent variables of the dataset.
    y : pd.Series
        Target vector corresponding to `X`.
    logger : Logger
        Logger instance used to record a debug message with details
        about the row removal operation.

    Returns
    -------
    X_clean : pd.DataFrame
        Feature matrix after duplicate rows were removed and indices reset.
    y_clean : pd.Series
        Target vector aligned with `X_clean`, with duplicates removed and indices reset.
    step_params : Dict[str, str]
        Metadata containing a human-readable description of the operation,
        including the number of dropped duplicates and their row indices if applicable.

    Notes
    -----
    - Only rows that have identical feature values in `X` *and* identical target values in `y`
        are considered duplicates and removed.
    - The first occurrence of each unique sample is retained (`keep="first"`).
    - The logger will produce a debug-level message describing the outcome.
    """
    num_rows_before: int = X.shape[0]

    # Concatenate X and y to consider both features and target in duplicate detection
    combined = X.copy()
    combined["_target"] = y.copy()

    # Identify duplicate rows (all except the first occurrence)
    duplicate_mask: pd.Series = combined.duplicated(keep="first")

    # Get index labels of duplicate rows that will be dropped
    dropped_indices: list[Any] = X.index[duplicate_mask].tolist()

    # Drop duplicate rows from X and y
    X_clean: pd.DataFrame = X.loc[~duplicate_mask].reset_index(drop=True)
    y_clean: pd.Series = y.loc[~duplicate_mask].reset_index(drop=True)

    # Correct row count after dropping duplicates
    num_rows_after: int = X_clean.shape[0]

    description = (
        f"{num_rows_before - num_rows_after} duplicate rows have been dropped."
    )
    if num_rows_after < num_rows_before:
        description += f"\nDropped rows with indices: {dropped_indices}"

    meta_data = {"Description": description}

    logger.debug(msg=f"[GREEN]- {description}")

    return X_clean, y_clean, meta_data


def drop_strings(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit: bool,
    step_params: Dict[str, Any],
    target_aware: bool = True,
    logger: Logger,
    step_outputs: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Drop any feature columns in training, validation, test, and optional test datasets
    that are detected as string dtype.

    This is often done to remove unprocessable text columns before modeling.
    """
    # Identify columns with string dtype (including object dtype with strings)
    if fit:
        drop_cols = []
        max_unique = min(20, max(10, int(0.01 * len(y))))
        for col in X.columns:
            if X[col].dtype in ["string", "category", "object"]:
                if len(X[col].unique()) > max_unique:
                    drop_cols.append(col)
        X.drop(columns=drop_cols, inplace=True)
        logger.debug(f"[GREEN]- dropping {drop_cols} as they are strings")
        step_params["drop_cols"] = drop_cols
        return X, y, step_params
    else:
        drop_cols = step_params["drop_cols"]
        X.drop(columns=drop_cols, inplace=True)
        return X, y, step_params


def drop_duplicate_columns(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    fit: bool,
    step_params: Dict[str, Any],
    target_aware: bool = True,
    logger: Logger,
    step_outputs: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Identifies and removes duplicate columns in the training features `self.X_train`
    by comparing hash sums of each column. Drops the duplicate columns from all
    datasets (train, validation, test, and optional test set).

    Logs the list of dropped columns or 'None' if no duplicates were found.
    """
    if fit:
        # Step 1: Compute hash sum per column to detect duplicates
        hashes: pd.DataFrame = X.apply(
            lambda col: pd.util.hash_pandas_object(col, index=False).sum()
        )

        # Step 2: Identify duplicated hashes (i.e., duplicate columns)
        duplicated_mask: pd.Series = hashes.duplicated()
        duplicate_columns: pd.Index[str] = X.columns[duplicated_mask]
        X = X.drop(columns=duplicate_columns)
        step_params = {"duplicate_columns": duplicate_columns}
        # Log the duplicate columns (if any)
        logger.debug(
            msg=f"[GREEN]- Duplicate columns to be dropped: {list(duplicate_columns) if len(duplicate_columns) > 0 else 'None'}"
        )
    else:
        X = X.drop(columns=step_params["duplicate_columns"])
    return X, y, step_params


def drop_constant_columns(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    fit: bool,
    step_params: Dict[str, Any],
    target_aware: bool = True,
    logger: Logger,
    step_outputs: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Detects columns in training features `self.X_train` with constant values
    (only one unique value) and drops them from all datasets (train, val, test,
    and optional test set).

    Logs which columns were dropped or 'None' if none were found.
    """
    if fit:
        # Vectorized approach: find columns where number of unique values is 1
        nunique_per_col: pd.Series = X.nunique()
        constant_columns: list[Any] = nunique_per_col[
            nunique_per_col == 1
        ].index.tolist()

        logger.debug(
            f"[GREEN]- Constant columns to be dropped: {constant_columns if len(constant_columns) > 0 else 'None'}"
        )

        # Drop constant columns from train, test, and val sets
        X = X.drop(columns=constant_columns)
        step_params = {"constant_columns": constant_columns}
    else:
        X = X.drop(columns=step_params["constant_columns"])
    return X, y, step_params
