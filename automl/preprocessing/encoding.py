# internal imports
from library import Logger


# external imports

import numpy as np
import pandas as pd


from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)

from typing import Tuple, Optional, Dict, Any
import warnings


def encode_target(
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
    Encode the target variables (y_train, y_val, y_test) if they are non-numeric.

    Uses LabelEncoder to convert string targets to numeric labels.
    Converts boolean targets to integers.

    No operation if targets are already numeric.
    """
    if fit:
        le = LabelEncoder()
        if not pd.api.types.is_numeric_dtype(y):
            # Fit on y_train
            le.fit(y)
            # Transform y_train, y_test, y_val using the same encoder
            y = pd.Series(
                le.transform(y),  # type: ignore
                index=y.index,
                name=y.name,
            )
            step_params["le"] = le
            step_params["transform"] = False
            step_params["boolean"] = False
        elif pd.api.types.is_bool_dtype(y) or set(y.dropna().unique()) <= {
            0,
            1,
        }:
            y = y.astype(dtype=int)
            step_params["le"] = None
            step_params["boolean"] = True
            step_params["transform"] = False
        else:
            step_params["le"] = None
            step_params["transform"] = True
            step_params["boolean"] = False
        return X, y, step_params
    else:
        if y is not None:
            if step_params["le"] is not None:
                le = step_params["le"]
                y = pd.Series(
                    le.transform(y),  # type: ignore
                    index=y.index,
                    name=y.name,
                )
            elif step_params["boolean"]:
                y = y.astype(int)
        return X, y, step_params


def auto_encode_features(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit: bool,
    step_params: Dict[str, Any],
    target_aware: bool = True,
    logger: Logger,
    step_outputs: Dict[str, Any],
    max_unique_for_categorical: int = 15,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Automatically encode categorical features in training, validation, test, and optional test datasets.

    Encoding schemes:
        - Boolean columns converted to int (0/1).
        - Categorical dtype or object columns encoded with OneHotEncoder (drop='first', handle_unknown='ignore').
        - Numeric columns with low unique values:
            * Encoded with OrdinalEncoder if target mean monotonically varies with feature.
            * Otherwise with OneHotEncoder.
        - Continuous numeric columns left unchanged.

    Args:
        max_unique_for_categorical (int): Maximum unique values in a numeric column to consider encoding as categorical.

    Returns:
        dict[str, OneHotEncoder | OrdinalEncoder]: Mapping from column name to its encoder used.
    """
    # suppress warnings during encoding when new categories are found in val/test/df_test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fit:
            encoders = {}
            transformed_train = X.copy()
            df = X.copy()

            for col in df.columns:
                col_data = df[col]
                unique_vals = col_data.nunique()
                dtype = col_data.dtype

                # Boolean: convert to int (0/1)
                if pd.api.types.is_bool_dtype(dtype):
                    transformed_train[col] = col_data.astype(int)
                    continue

                # Category dtype: nominal categorical -> OneHotEncoder with handle_unknown='ignore'
                elif isinstance(dtype, pd.CategoricalDtype):
                    logger.debug(
                        f"[GREEN]- Applying OneHotEncoder on column '{col}'"
                    )
                    encoder = OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    )
                    encoded_train = encoder.fit_transform(
                        np.asarray(col_data).reshape(-1, 1)
                    )
                    cats = encoder.categories_[0]
                    cols_one_hot = [f"{col}_{cat}" for cat in cats[1:]]  # type: ignore
                    encoded_train_df = pd.DataFrame(
                        encoded_train, columns=cols_one_hot, index=df.index
                    )
                    transformed_train = transformed_train.drop(
                        columns=[col]
                    ).join(encoded_train_df)
                    encoders[col] = encoder
                    continue

                # String/object dtype: treat as nominal categorical with OneHotEncoder and handle_unknown='ignore'
                elif pd.api.types.is_object_dtype(dtype):
                    logger.debug(
                        f"[GREEN]- Applying OneHotEncoder on column '{col}'"
                    )
                    temp_cat = col_data.astype("category")
                    encoder = OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    )
                    encoded_train = encoder.fit_transform(
                        np.asarray(temp_cat.values).reshape(-1, 1)
                    )
                    cats = encoder.categories_[0]
                    cols_one_hot = [f"{col}_{cat}" for cat in cats[1:]]  # type: ignore
                    encoded_train_df = pd.DataFrame(
                        encoded_train, columns=cols_one_hot, index=df.index
                    )
                    transformed_train = transformed_train.drop(
                        columns=[col]
                    ).join(encoded_train_df)
                    encoders[col] = encoder
                    continue

                # Numeric dtype with low cardinality: one-hot or ordinal encoding with unknown category handling
                elif pd.api.types.is_numeric_dtype(dtype):
                    if unique_vals <= max_unique_for_categorical:
                        if y is not None:
                            df_combined = df.copy()

                            df_combined[y.name] = y

                            means = df_combined.groupby(col)[  # type: ignore
                                y.name
                            ].mean()
                            if (
                                means.is_monotonic_increasing
                                or means.is_monotonic_decreasing
                            ):
                                # OrdinalEncoder with unknown category handling
                                encoder = OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                )
                                transformed_train[[col]] = (
                                    encoder.fit_transform(
                                        np.asarray(col_data.values).reshape(
                                            -1, 1
                                        )
                                    )
                                )
                                logger.debug(
                                    f"[GREEN]- Applying OrdinalEncoder with unknown category handling on column '{col}'"
                                )
                                encoders[col] = encoder
                            else:
                                # OneHotEncoder with handle_unknown='ignore'
                                encoder = OneHotEncoder(
                                    drop="first",
                                    sparse_output=False,
                                    handle_unknown="ignore",
                                )
                                encoded_train = encoder.fit_transform(
                                    np.asarray(col_data.values).reshape(-1, 1)
                                )
                                cols_one_hot = [
                                    f"{col}_{cat}"
                                    for cat in encoder.categories_[0][1:]  # type: ignore
                                ]
                                encoded_train_df = pd.DataFrame(
                                    encoded_train,
                                    columns=cols_one_hot,
                                    index=df.index,
                                )
                                transformed_train = transformed_train.drop(
                                    columns=[col]
                                ).join(encoded_train_df)

                                logger.debug(
                                    f"[GREEN]- Applying OneHotEncoder with unknown category handling on column {col}"
                                )
                                encoders[col] = encoder
                        else:
                            encoder = OneHotEncoder(
                                drop="first",
                                sparse_output=False,
                                handle_unknown="ignore",
                            )
                            encoded_train = encoder.fit_transform(
                                np.asarray(col_data.values).reshape(-1, 1)
                            )
                            cols_one_hot = [
                                f"{col}_{cat}" for cat in encoder.categories_[0][1:]  # type: ignore
                            ]
                            encoded_train_df = pd.DataFrame(
                                encoded_train,
                                columns=cols_one_hot,
                                index=df.index,
                            )
                            transformed_train = transformed_train.drop(
                                columns=[col]
                            ).join(encoded_train_df)

                            logger.debug(
                                f"[GREEN]- Applying OneHotEncoder with unknown category handling on column '{col}'"
                            )
                            encoders[col] = encoder
                    else:
                        # Leave continuous numeric unchanged
                        pass
            added_columns = list(
                set(transformed_train.columns) - set(X.columns)
            )
            X = transformed_train
            step_params["encoders"] = encoders
            step_params["added_columns"] = added_columns
            return X, y, step_params
        else:
            encoders = step_params["encoders"]
            drop_cols = []
            transformed_X = X.copy()
            for col, encoder in encoders.items():
                if isinstance(encoder, OrdinalEncoder):
                    col_data = transformed_X[col]
                    transformed_X[[col]] = encoder.fit_transform(
                        np.asarray(col_data).reshape(-1, 1)
                    )
                else:
                    encoded_X = encoder.transform(X[col].values.reshape(-1, 1))
                    cats = encoder.categories_[0]
                    cols_enc = [f"{col}_{cat}" for cat in cats[1:]]
                    encoded_X_df = pd.DataFrame(
                        encoded_X,  # type: ignore
                        columns=cols_enc,
                        index=X.index,
                    )
                    transformed_X = transformed_X.drop(columns=[col]).join(
                        encoded_X_df
                    )
                    drop_cols.append(col)
            return transformed_X, y, step_params
