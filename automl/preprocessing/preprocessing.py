# internal imports
from library import Logger
from .target import (
    is_target_categorical,
    encode_target,
    standardize_target,
)
from .general import drop_duplicate_rows, skip_outliers, detect_dataset_type

# external imports
import pandas as pd
from typing import Dict, Any, Tuple


def preprocess(
    X: pd.DataFrame, y: pd.Series, logger: Logger
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Dict[str, Any]]]:
    X_prepro = X.copy()
    y_prepro = y.copy()
    meta_data = {}
    # first drop all duplicate rows
    X_prepro, y_prepro, meta_data["drop_duplicate_rows"] = drop_duplicate_rows(
        X=X_prepro, y=y_prepro, logger=logger
    )
    # find out what kind of dataset this is, based on target y
    target_categorical = is_target_categorical(target=y)
    if target_categorical:
        y_prepro, meta_data["encode_target"] = encode_target(target=y_prepro)
    else:
        y_prepro, meta_data["standardize_target"] = standardize_target(
            target=y_prepro, logger=logger
        )
    meta_data["skip_outliers"] = skip_outliers(target=y_prepro)
    meta_data["dataset_type"] = detect_dataset_type(target=y_prepro)
    return X_prepro, y_prepro, meta_data
