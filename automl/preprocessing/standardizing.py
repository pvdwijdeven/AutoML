from library import Logger

# external imports

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
)
from typing import Tuple, Optional, Dict, Any


def normalize_columns(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit: bool,
    step_params: Dict[str, Any] = {},
    target_aware: bool = True,
    logger: Logger,
    step_outputs: Dict[str, Any] = {},
    skewness_threshold: float = 0.75,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Apply power transform (Yeo-Johnson) to skewed numeric columns and standard scale all numeric columns.

    Columns with absolute skewness higher than `skewness_threshold` are transformed.
    Scaling is fit on training set and applied to validation, test, and optional test.

    Args:
        skewness_threshold (float): Skewness level beyond which to apply Yeo-Johnson transform.
    """
    if fit:
        added_columns = step_outputs["auto_encode_features"]["added_columns"]
        logger.debug("[GREEN]- Normalizing")
        for column_name in X.columns:
            if column_name in added_columns:
                # skip encoded columns
                continue
            if not is_numeric_dtype(X[column_name]):
                continue
            skewness_value = X[column_name].skew()
            step_params[column_name] = {}
            try:
                skewness_float = float(skewness_value)  # type: ignore
            except (TypeError, ValueError):
                continue  # skip columns where skewness cannot be converted to float
            if abs(skewness_float) > skewness_threshold:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                # Fit on train column reshaped as 2D array
                X_train_col = X[
                    [column_name]
                ]  # keeps DataFrame format for transform
                pt.fit(X_train_col)

                # Transform all three sets' column
                X[column_name] = pt.transform(X_train_col)
                step_params[column_name]["pt"] = pt
                logger.debug(
                    f"- Skewness found for {column_name}, yeo-johnson transormer applied"
                )
            scaler = StandardScaler()
            train_col = X[[column_name]]
            scaler.fit(train_col)
            step_params[column_name]["scaler"] = scaler
            # Transform returns 2D array, extract to 1D to assign safely
            X.loc[:, column_name] = (
                scaler.transform(train_col).flatten().astype(np.float64)
            )
        return X, y, step_params
    else:
        for column_name, scalers in step_params.items():
            if "pt" in step_params[column_name]:
                pt = step_params[column_name]["pt"]
                X_train_col = X[[column_name]]
                X[column_name] = pt.transform(X_train_col)
            if "scaler" in step_params[column_name]:
                scaler = step_params[column_name]["scaler"]
                X_train_col = X[[column_name]]
                X.loc[:, column_name] = (
                    scaler.transform(X_train_col).flatten().astype(np.float64)
                )
        return X, y, step_params
