from library import Logger

# external imports

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
)
from typing import Tuple, Optional, Dict, Any, Self


class TargetTransformer:
    """
    Transforms the target variable using Yeo-Johnson power transform (to reduce skewness)
    followed by standard scaling (zero mean, unit variance). Supports inverse transformation.

    Attributes:
        pt (PowerTransformer): PowerTransformer instance using Yeo-Johnson method.
        scaler (StandardScaler): StandardScaler instance to scale the transformed target.
    """

    def __init__(self) -> None:
        """
        Initializes the TargetTransformer with PowerTransformer and StandardScaler.
        """
        self.pt: PowerTransformer = PowerTransformer(
            method="yeo-johnson", standardize=False
        )
        self.scaler: StandardScaler = StandardScaler()

    def fit(self, y_train: pd.Series | np.ndarray) -> Self:
        """
        Fits the power transformer and scaler on the training target data.

        Args:
            y_train (pd.Series | np.ndarray): Training target values.

        Returns:
            Self: The fitted transformer instance.
        """
        if isinstance(y_train, np.ndarray):
            y_train_reshaped = y_train.reshape(-1, 1)
        else:
            y_train_reshaped = np.asarray(y_train).reshape(-1, 1)

        y_transformed = self.pt.fit_transform(y_train_reshaped)
        self.scaler.fit(y_transformed)
        return self

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """
        Transforms the target data using the fitted power transformer and scaler.

        Args:
            y (pd.Series | np.ndarray): Target values to transform.

        Returns:
            np.ndarray: Transformed and scaled target data as 1D array.
        """
        if isinstance(y, np.ndarray):
            y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = np.asarray(y).reshape(-1, 1)
        y_transformed = self.pt.transform(y_reshaped)
        y_scaled = self.scaler.transform(y_transformed)
        return y_scaled.flatten()

    def inverse_transform(self, y_scaled: pd.Series | np.ndarray) -> np.ndarray:
        """
        Inverse transforms scaled target data back to the original scale.

        Args:
            y_scaled (pd.Series | np.ndarray): Scaled target values.

        Returns:
            np.ndarray: Original scale target values as 1D array.
        """
        if isinstance(y_scaled, pd.Series):
            y_scaled = np.asanyarray(y_scaled)
        y_transformed = self.scaler.inverse_transform(y_scaled.reshape(-1, 1))
        y_original = self.pt.inverse_transform(y_transformed)
        return y_original.flatten()


def normalize_columns(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit: bool,
    step_params: Dict[str, Any],
    target_aware: bool = True,
    logger: Logger,
    step_outputs: Dict[str, Any],
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
        for column_name in step_params:
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


def standardize_target(
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
    Fit a TargetTransformer using Yeo-Johnson transform and standard scaling on y_train,
    then transform y_train, y_val, and y_test targets correspondingly.

    Stores the target_transformer object for later inverse transformations if needed.
    """
    if fit:
        if step_outputs["encode_target"]["transform"]:
            # 1. Fit transformer on training target
            target_transformer = TargetTransformer()
            target_transformer.fit(y)

            # 2. Transform target in train, val, test sets
            y = pd.Series(
                target_transformer.transform(y),
                index=y.index,
                name=y.name,
            )
            step_params["target_transformer"] = target_transformer
        else:
            step_params["target_transformer"] = None
        return X, y, step_params
    else:
        # target_transformer = step_params["target_transformer"]
        # if target_transformer is not None:
        #     y = pd.Series(
        #         target_transformer.transform(y),
        #         index=y.index,
        #         name=y.name,
        #     )
        return X, y, step_params
