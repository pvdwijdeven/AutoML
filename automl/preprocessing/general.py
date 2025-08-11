from library import Logger

import pandas as pd
from typing import Optional, Tuple, Dict, Any


def drop_duplicate_rows(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    fit: bool,
    step_params: Dict[str, Any] = {},
    target_aware: bool = True,
    logger: Logger,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Removes duplicate rows from the original training dataframe `self.eda.df_train`.
    Logs the number of duplicates dropped and their indices.
    """
    if fit:
        num_rows_before: int = X.shape[0]

        # Identify duplicate rows (all except the first occurrence)
        duplicate_mask: pd.Series = X.duplicated(
            keep="first"
        )  # True for duplicates
        # Get index labels of duplicate rows that will be dropped
        dropped_indices: list[Any] = X.index[duplicate_mask].tolist()
        # Drop duplicate rows
        X_clean: pd.DataFrame = X.loc[~duplicate_mask].reset_index(drop=True)
        if target_aware and y is not None:
            y_clean = y.loc[~duplicate_mask].reset_index(drop=True)
        else:
            y_clean: pd.Series | None = y
        num_rows_after: int = X.shape[0]
        if num_rows_after < num_rows_before:
            logger.info(
                msg=f"[GREEN]- Duplicate rows to be dropped: {num_rows_before - num_rows_after}"
            )
            logger.debug(
                msg=f"[GREEN]  Dropped rows with indices: {dropped_indices}"
            )
        else:
            logger.info(msg="[GREEN]- Duplicate rows to be dropped: None")
        return X_clean, y_clean, step_params
    else:
        return X, y, step_params


def drop_duplicate_columns(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    fit: bool,
    step_params: Dict[str, Any] = {},
    target_aware: bool = True,
    logger: Logger,
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
        logger.info(
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
    step_params: Dict[str, Any] = {},
    target_aware: bool = True,
    logger: Logger,
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

        logger.info(
            f"[GREEN]- Constant columns to be dropped: {constant_columns if len(constant_columns) > 0 else 'None'}"
        )

        # Drop constant columns from train, test, and val sets
        X = X.drop(columns=constant_columns)
        step_params = {"constant_columns": constant_columns}
    else:
        X = X.drop(columns=step_params["constant_columns"])
    return X, y, step_params
