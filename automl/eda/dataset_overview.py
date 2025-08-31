# Third-party imports
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pydantic import BaseModel

from .column_analysis import ColumnInfo


class DatasetInfo(BaseModel):
    number_features: int
    number_constant_features: int
    perc_constant_features: float
    number_duplicate_features: int
    perc_duplicate_features: float
    number_empty_features: int
    perc_empty_features: float
    feature_types: dict[str, int]
    number_samples: int
    number_duplicate_samples: int
    perc_duplicate_samples: float
    number_empty_samples: int
    perc_empty_samples: float
    table_size: str  # example: "14 x 891"
    number_cells: int
    memory: str  # example: 410.39 KB
    number_missing: int
    perc_missing: float
    target: str
    target_type: str
    dataset_type: str


def find_duplicate_columns(X_train: DataFrame):
    """
    Find columns in the DataFrame that have duplicate values with other columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary where keys are column names, and values are lists of duplicated column names.
    """
    duplicates_dict: dict[str, list[str]] = {}

    columns = X_train.columns
    n = len(columns)

    for i in range(n):
        cur_col = columns[i]
        duplicates = []
        for j in range(i + 1, n):
            other_col = columns[j]
            # Check if all values in cur_col equal those in other_col
            if X_train[cur_col].equals(other=X_train[other_col]):
                duplicates.append(other_col)
        duplicates_dict[cur_col] = duplicates

    return duplicates_dict


def detect_dataset_type(
    target: Series,
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


def analyse_dataset(
    X_train: DataFrame,
    column_info: dict[str, ColumnInfo],
    dict_duplicates: dict[str, list[str]],
    y_train: Series,
) -> DatasetInfo:
    number_features = len(X_train.columns)
    constant_features = sum(
        [column_info[col].is_constant for col in column_info]
    )
    number_duplicate_features = sum(
        [len(dict_duplicates[x]) > 0 for x in dict_duplicates]
    )
    number_empty_features = sum(
        [int(X_train[column].isnull().all()) for column in X_train.columns]
    )
    duplicate_samples = (
        pd.util.hash_pandas_object(X_train.round(5), index=False)
        .duplicated()
        .sum()
    )
    number_samples = X_train.shape[0]
    number_empty_rows = (X_train.isnull().sum(axis=1) == X_train.shape[1]).sum()
    number_missing = X_train.isnull().sum().sum()
    return DatasetInfo(
        number_features=number_features,
        number_constant_features=constant_features,
        perc_constant_features=constant_features / number_features * 100,
        number_duplicate_features=number_duplicate_features,
        perc_duplicate_features=number_duplicate_features
        / number_features
        * 100,
        number_empty_features=number_empty_features,
        perc_empty_features=number_empty_features / number_features * 100,
        feature_types={},
        number_samples=number_samples,
        number_duplicate_samples=duplicate_samples,
        perc_duplicate_samples=duplicate_samples / number_samples * 100,
        number_empty_samples=number_empty_rows,
        perc_empty_samples=number_empty_rows / number_samples * 100,
        table_size=f"{number_features} x {number_samples}",
        number_cells=number_features * number_samples,
        memory=f"{(X_train.memory_usage(deep=True).sum() / 1024):.3f} kB",
        number_missing=number_missing,
        perc_missing=number_missing / (number_features * number_samples) * 100,
        target=column_info["target"].column_name,
        target_type=column_info["target"].proposed_type,
        dataset_type=detect_dataset_type(target=y_train),
    )
