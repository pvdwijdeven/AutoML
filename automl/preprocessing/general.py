# Standard library imports
from typing import Any, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd

# Local application imports
from automl.library import Logger


def detect_dataset_type(
    target: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> str:
    """
    Detect the type of supervised learning problem based on the characteristics of the target variable.

    The function inspects the target (labels) and categorizes it into one of:
    - "multi_label_classification": when target is a DataFrame with multiple binary columns.
    - "regression": when numeric with many unique values (> 20).
    - "imbalanced_binary_classification": when binary with high imbalance (minority class <= 5%).
    - "binary_classification": when binary and reasonably balanced.
    - "multi_class_classification": when categorical or numeric with few discrete levels.
    - "ordinal_regression": when integer-valued, contiguous classes, between 3 and 20 categories.
    - "unknown": fallback if none of the above matches.

    Parameters
    ----------
    target : Union[pd.DataFrame, pd.Series, np.ndarray]
        The target variable(s). Can be a DataFrame (multi-output),
        a Series (single-output), or a NumPy ndarray.

    Returns
    -------
    str
        The detected dataset type. One of:
        [
            "multi_label_classification",
            "regression",
            "imbalanced_binary_classification",
            "binary_classification",
            "multi_class_classification",
            "ordinal_regression",
            "unknown",
        ]
    """
    # Convert to pandas object for convenience
    if isinstance(target, (pd.DataFrame, pd.Series)):
        target_df = target
    else:  # Handle NumPy array
        if target.ndim == 1:
            target_df = pd.Series(data=target)
        elif target.ndim == 2:
            target_df = pd.DataFrame(data=target)
        else:
            raise ValueError("Target must be 1D or 2D array-like")

    # Multi-label case: DataFrame with multiple columns
    if isinstance(target_df, pd.DataFrame) and target_df.shape[1] > 1:
        unique_vals = pd.unique(target_df.values.ravel())
        if set(unique_vals).issubset({0, 1}):
            return "multi_label_classification"
        return "multi_label_classification"

    # 1D case: Series
    target_series = (
        target_df if isinstance(target_df, pd.Series) else target_df.iloc[:, 0]
    )
    is_numeric = pd.api.types.is_numeric_dtype(target_series)

    unique_vals = target_series.dropna().unique()
    n_unique = len(unique_vals)

    # Numeric continuous => regression
    if is_numeric and n_unique > 20:
        return "regression"

    # Binary classification
    if n_unique == 2:
        counts = target_series.value_counts(normalize=True)
        imbalance_threshold = 0.05
        minority_ratio = counts.min()
        return (
            "imbalanced_binary_classification"
            if minority_ratio <= imbalance_threshold
            else "binary_classification"
        )

    # Multi-class or ordinal case
    if not is_numeric:
        return "multi_class_classification"

    unique_vals_sorted = np.sort(unique_vals)
    diffs = np.diff(unique_vals_sorted)

    if np.all(diffs == 1) and np.all(
        unique_vals_sorted == unique_vals_sorted.astype(int)
    ):
        if 3 <= n_unique <= 20:
            return "ordinal_regression"
        return "multi_class_classification"

    if n_unique <= 20:
        return "multi_class_classification"

    # Fallback
    return "unknown"


def skip_outliers(target: pd.Series) -> dict[str, bool]:
    """
    Check whether extremely rare classes exist in a categorical target variable.

    If any class in the target distribution has frequency < 1% of the dataset
    and the total number of unique classes is <= 5, the function flags
    that outliers should be skipped.

    Parameters
    ----------
    target : pd.Series
        Target values (assumed categorical for this heuristic).

    Returns
    -------
    dict[str, bool]
        dictionary with key "skip_outliers" set to True if rare
        classes are detected, otherwise False.
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
) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
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
    step_params : dict[str, str]
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
    y: Optional[pd.Series],
    *,
    fit: bool,
    step_params: dict[str, Any],
    logger: Logger,
    meta_data: dict[str, Any],
) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[dict[str, Any]]]:
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
    else:
        string_columns = step_params.get("drop_cols", [])
        if string_columns:
            logger.debug(
                f"[GREEN]- Dropping string-like columns: {string_columns}"
            )
            X = X.drop(columns=string_columns)

    return X, y, step_params


def drop_duplicate_columns(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    *,
    fit: bool,
    step_params: dict[str, Any],
    logger: Logger,
    meta_data: dict[str, Any],
) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[dict[str, Any]]]:
    """
    Drops duplicate columns in the dataset.

    During the *fit* stage, columns with identical content are detected
    using hash-based comparison. Duplicate columns are stored in step_params.
    During the *transform* stage, the recorded duplicate columns are dropped.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : Optional[pd.Series]
        Target values (unchanged).
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
        hashes = X.apply(
            lambda col: pd.util.hash_pandas_object(col, index=False).sum()
        )
        duplicated_mask = hashes.duplicated()
        duplicate_columns = X.columns[duplicated_mask]
        step_params["duplicate_columns"] = duplicate_columns
    else:
        duplicate_columns = step_params.get("duplicate_columns", [])
        if len(duplicate_columns) > 0:
            logger.debug(
                f"[GREEN]- Dropping duplicate columns: {list(duplicate_columns)}"
            )
            X = X.drop(columns=duplicate_columns)
    return X, y, step_params


def drop_constant_columns(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    *,
    fit: bool,
    step_params: dict[str, Any],
    logger: Logger,
    meta_data: dict[str, Any],
) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[dict[str, Any]]]:
    """
    Drops constant columns (columns with only one unique value).

    During the *fit* stage, columns with a single unique value are identified
    and stored in step_params. During the *transform* stage, these columns
    are dropped from the dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : Optional[pd.Series]
        Target values (unchanged).
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
        nunique_per_col = X.nunique(dropna=True)
        constant_columns = nunique_per_col[nunique_per_col == 1].index.tolist()
        step_params["constant_columns"] = constant_columns
    else:
        constant_columns = step_params.get("constant_columns", [])
        if constant_columns:
            logger.debug(
                f"[GREEN]- Dropping constant columns: {constant_columns}"
            )
            X = X.drop(columns=constant_columns)

    return X, y, step_params
