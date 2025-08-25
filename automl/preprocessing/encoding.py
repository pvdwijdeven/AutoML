# Standard library imports
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Local application imports
from automl.library import Logger


def auto_encode_features(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    *,
    fit: bool,
    step_params: Dict[str, Any],
    logger: Logger,
    meta_data: Dict[str, Any],
    max_unique_for_categorical: int = 15,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Automatically encode categorical and boolean features in the dataset.

    Fit mode:
    - Fit appropriate encoders per column (OneHotEncoder, OrdinalEncoder, or boolean to int conversion).
    - Does NOT transform X during fit (no data leakage).
    - Store fitted encoder objects and list of added encoded columns in step_params.

    Transform mode:
    - Apply fitted encoders to X, replacing original columns with encoded columns.
    - Boolean columns converted to int.
    - Maintains consistency with fitted encoding scheme, ignoring unseen categories.

    Encoding Strategy:
    - Boolean columns: convert to integer 0/1.
    - Categorical dtype: OneHotEncoder with handle_unknown='ignore' and drop='first'.
    - Object dtype: treated as categorical and OneHotEncoded similarly.
    - Numeric dtype with low unique values (<= max_unique_for_categorical):
      - If target y provided and mean target per category is monotonic (increasing or decreasing),
        use OrdinalEncoder with unknown category handling.
      - Otherwise, use OneHotEncoder as above.
    - Numeric dtype with high unique values: left unchanged.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.
    y : Optional[pd.Series]
        Target values, used only for ordinal encoding heuristic.
    fit : bool
        Whether to fit encoders (True) or transform using saved encoders (False).
    step_params : Dict[str, Any]
        Stores encoders and metadata across fit/transform stages.
    logger : Logger
        Logger for debug/info messages.
    meta_data : Dict[str, Any]
        Metadata dictionary (unused currently).
    max_unique_for_categorical : int, default=15
        Maximum unique values to consider numeric feature as categorical.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]
        Transformed feature DataFrame (only on transform), unchanged target,
        and updated step_params containing encoders and added columns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if fit:
            encoders: Dict[str, Any] = {}
            added_columns: List[str] = []
            # Do not transform X during fit; only fit encoders

            for col in X.columns:
                col_data = X[col]
                unique_vals = col_data.nunique()
                dtype = col_data.dtype

                # Boolean: no fitting needed, just convert in transform
                if ptypes.is_bool_dtype(dtype):
                    encoders[col] = "bool_to_int"
                    logger.debug(
                        f"[GREEN]- Registered boolean to int conversion for column '{col}'"
                    )
                    continue

                # Categorical dtype: fit OneHotEncoder
                if isinstance(dtype, pd.CategoricalDtype):
                    logger.debug(
                        f"[GREEN]- Fitting OneHotEncoder for categorical column '{col}'"
                    )
                    encoder = OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    )
                    encoder.fit(np.asarray(col_data).reshape(-1, 1))
                    encoders[col] = encoder
                    continue

                # Object dtype: treat as categorical
                if ptypes.is_object_dtype(dtype):
                    logger.debug(
                        f"[GREEN]- Fitting OneHotEncoder for object-type column '{col}'"
                    )
                    temp_cat = col_data.astype("category")
                    encoder = OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    )
                    encoder.fit(np.asarray(temp_cat.values).reshape(-1, 1))
                    encoders[col] = encoder
                    continue

                # Numeric with limited cardinality: decide ordinal or one-hot
                if ptypes.is_numeric_dtype(dtype):
                    if unique_vals <= max_unique_for_categorical:
                        if y is not None and y.name is not None:
                            df_combined = X.copy()
                            df_combined[y.name] = y
                            means = df_combined.groupby(col)[y.name].mean()  # type: ignore
                            if (
                                means.is_monotonic_increasing
                                or means.is_monotonic_decreasing
                            ):
                                # Ordinal encoder with unknown category handling
                                encoder = OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                )
                                encoder.fit(
                                    np.asarray(col_data.values).reshape(-1, 1)
                                )
                                encoders[col] = encoder
                                logger.debug(
                                    f"[GREEN]- Fitting OrdinalEncoder for monotonic numeric column '{col}'"
                                )
                                continue
                            # else fall through to one-hot below

                        # OneHotEncoder fallback for numeric categorical
                        encoder = OneHotEncoder(
                            drop="first",
                            sparse_output=False,
                            handle_unknown="ignore",
                        )
                        encoder.fit(np.asarray(col_data.values).reshape(-1, 1))
                        encoders[col] = encoder
                        logger.debug(
                            f"[GREEN]- Fitting OneHotEncoder for numeric categorical column '{col}'"
                        )
                        continue

                # Numeric with high uniqueness left unchanged (no encoder)
                logger.debug(
                    f"[GREEN]- Leaving continuous numeric column '{col}' unchanged"
                )

            step_params["encoders"] = encoders
            step_params["added_columns"] = added_columns
            return X, y, step_params

        else:  # transform
            encoders = step_params.get("encoders", {})
            transformed_X = X.copy()
            drop_cols: List[str] = []

            for col, encoder in encoders.items():
                if encoder == "bool_to_int":
                    # Convert boolean column to int
                    if col in transformed_X.columns and ptypes.is_bool_dtype(
                        transformed_X[col].dtype
                    ):
                        transformed_X[col] = transformed_X[col].astype(int)
                        logger.debug(
                            f"[GREEN]- Converted boolean column '{col}' to int during transform"
                        )
                    continue

                if isinstance(encoder, OrdinalEncoder):
                    if col in transformed_X.columns:
                        col_data = transformed_X[col]
                        transformed_X[[col]] = encoder.transform(
                            np.asarray(col_data).reshape(-1, 1)
                        )
                    continue

                if isinstance(encoder, OneHotEncoder):
                    if col in transformed_X.columns:
                        encoded = encoder.transform(
                            transformed_X[col].values.reshape(-1, 1)  # type: ignore
                        )
                        cats = encoder.categories_[0]
                        encoded_cols = [f"{col}_{cat}" for cat in cats[1:]]  # type: ignore
                        encoded_df = pd.DataFrame(
                            encoded,  # type: ignore
                            columns=encoded_cols,
                            index=transformed_X.index,
                        )
                        transformed_X = transformed_X.drop(columns=[col]).join(
                            encoded_df
                        )
                        drop_cols.append(col)
                    continue

            return transformed_X, y, step_params
