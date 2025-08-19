from library import Logger

# external imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
)
from typing import Self, Dict, Any, Tuple
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
)


def is_target_categorical(
    target: pd.Series, unique_threshold: int = 10
) -> bool:
    """
    Determine if a target variable should be considered categorical based on unique value count.

    Parameters:
    - target: pd.Series, the target variable
    - unique_threshold: int, maximum number of unique values to consider target categorical (default=10)

    Returns:
    - bool: True if target is categorical, False otherwise
    """
    unique_values = target.nunique()
    return unique_values <= unique_threshold


def encode_target(target: pd.Series) -> tuple[pd.Series, Dict[str, Any]]:
    """
    Encode a pandas Series categorical target using LabelEncoder and keep result as pandas Series.

    Parameters:
    - y: pandas Series, the target variable

    Returns:
    - y_encoded: pandas Series of encoded target
    - le: fitted LabelEncoder instance
    """
    le = LabelEncoder()
    y_encoded_array = le.fit_transform(y=target)
    y_encoded = pd.Series(y_encoded_array, index=target.index, name=target.name)  # type: ignore
    meta_data = {
        "encoder": le,
        "description": "Target has been encoded with LabelEncoder",
        "conversion": {
            label: idx for idx, label in enumerate(iterable=le.classes_)
        },
    }
    return y_encoded, meta_data


def decode_target(y_encoded, label_encoder) -> tuple[pd.Series, Dict[str, Any]]:
    """
    Decode a pandas Series encoded target back to original categories using a fitted LabelEncoder.

    Parameters:
    - y_encoded: pandas Series of encoded target
    - label_encoder: fitted LabelEncoder instance

    Returns:
    - pandas Series with original categories
    """
    decoded_array = label_encoder.inverse_transform(y_encoded)
    y_decoded = pd.Series(
        decoded_array, index=y_encoded.index, name=y_encoded.name
    )
    meta_data = {
        "encoder": label_encoder,
        "description": "Target has been decoded with LabelEncoder",
    }
    return y_decoded, meta_data


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

    def __repr__(self):
        return "TargetTransformer()"

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

    def inverse_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """
        Inverse transforms scaled target data back to the original scale.

        Args:
            y_scaled (pd.Series | np.ndarray): Scaled target values.

        Returns:
            np.ndarray: Original scale target values as 1D array.
        """
        if isinstance(y, pd.Series):
            y = np.asanyarray(y)
        y_transformed = self.scaler.inverse_transform(y.reshape(-1, 1))
        y_original = self.pt.inverse_transform(y_transformed)
        return y_original.flatten()


def standardize_target(
    target: pd.Series,
    logger: Logger,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Fit a TargetTransformer using Yeo-Johnson transform and standard scaling on y_train,
    then transform y_train, y_val, and y_test targets correspondingly.

    Stores the target_transformer object for later inverse transformations if needed.
    """
    # 1. Fit transformer on training target
    target_transformer = TargetTransformer()
    target_transformer.fit(y_train=target)

    # 2. Transform target in train, val, test sets
    target = pd.Series(
        data=target_transformer.transform(y=target),
        index=target.index,
        name=target.name,
    )
    meta_data = {"target_transformer": target_transformer}
    return target, meta_data
