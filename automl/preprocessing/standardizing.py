# Standard library imports
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Local application imports
from automl.library import Logger


def normalize_columns(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    *,
    fit: bool,
    step_params: Dict[str, Any],
    logger: Logger,
    meta_data: Dict[str, Any],
    skewness_threshold: float = 0.75,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Normalize numeric columns with optional power transformation if highly skewed.

    Columns that have been created by auto-encoding (from meta_data["auto_encode_features"]["added_columns"])
    are skipped. For each remaining numeric column:
    - If abs(skewness) > skewness_threshold, fit a Yeo-Johnson PowerTransformer.
    - Always fit a StandardScaler to normalize mean and variance.
    - All fitted transformers are saved in step_params so the same transformations
      are applied in transform stage.

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe.
    y : Optional[pd.Series]
        Target variable (unused here).
    fit : bool
        If True, fit normalization transformers and save state.
        If False, apply saved transformers.
    step_params : Dict[str, Any]
        Dictionary (per column) for storing fitted scalers and transformers as needed.
    logger : Logger
        Logger for debug output.
    meta_data : Dict[str, Any]
        Must contain "auto_encode_features" with "added_columns" list.
    skewness_threshold : float, default=0.75
        Absolute value above which skewed numeric columns get power transformed before scaling.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]
        Dataframe X (normalized), y (unchanged), and step_params (with fit state).

    Notes
    -----
    - If skewness cannot be evaluated or a column isn’t numeric, that column is skipped.
    - Original values are replaced in X, no new columns are added.
    """
    if fit:
        added_columns = meta_data.get("auto_encode_features", {}).get(
            "added_columns", []
        )
        for column_name in X.columns:
            if column_name in added_columns:
                continue  # Skip encoded columns
            if not is_numeric_dtype(X[column_name]):
                continue  # Skip non-numeric columns

            step_params[column_name] = {}
            # Evaluate skewness, skip if not convertible to float
            try:
                skewness_value = float(X[column_name].skew())  # type: ignore
            except (TypeError, ValueError):
                continue

            # If column is highly skewed, fit Yeo-Johnson PowerTransformer
            if abs(skewness_value) > skewness_threshold:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                X_col = X[[column_name]]
                pt.fit(X_col)
                step_params[column_name]["pt"] = pt

            # Always fit a StandardScaler
            scaler = StandardScaler()
            X_col = X[[column_name]]
            scaler.fit(X_col)
            step_params[column_name]["scaler"] = scaler

        return X, y, step_params

    else:
        logger.debug("[GREEN]- Normalizing")
        for column_name in step_params:
            # Power transformation if present
            if "pt" in step_params[column_name]:
                pt = step_params[column_name]["pt"]
                X_col = X[[column_name]]
                X[column_name] = pt.transform(X_col)
            # Standard scaling always
            if "scaler" in step_params[column_name]:
                scaler = step_params[column_name]["scaler"]
                X_col = X[[column_name]]
                X.loc[:, column_name] = (
                    scaler.transform(X_col).flatten().astype(np.float64)
                )
        return X, y, step_params
