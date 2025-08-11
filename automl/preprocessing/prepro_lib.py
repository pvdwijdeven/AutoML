from library import Logger

import pandas as pd
from typing import Optional


def drop_duplicate_rows(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    fit: bool,
    step_params: Optional[dict] = None,
    target_aware: bool = True,
    logger: Logger,
) -> tuple:
    """
    Removes duplicate rows from the original training dataframe `self.eda.df_train`.
    Logs the number of duplicates dropped and their indices.
    """
    if fit:
        num_rows_before = X.shape[0]

        # Identify duplicate rows (all except the first occurrence)
        duplicate_mask = X.duplicated(keep="first")  # True for duplicates
        # Get index labels of duplicate rows that will be dropped
        dropped_indices = X.index[duplicate_mask].tolist()
        # Drop duplicate rows
        X_clean = X.loc[~duplicate_mask].reset_index(drop=True)
        if target_aware and y is not None:
            y_clean = y.loc[~duplicate_mask].reset_index(drop=True)
        else:
            y_clean = y
        num_rows_after = X.shape[0]
        if num_rows_after < num_rows_before:
            logger.info(
                f"[GREEN]- Duplicate rows to be dropped: {num_rows_before - num_rows_after}"
            )
            logger.debug(
                f"[GREEN]  Dropped rows with indices: {dropped_indices}"
            )
        else:
            logger.info("[GREEN]- Duplicate rows to be dropped: None")
        return X_clean, y_clean, step_params
    else:
        return X, y, step_params
