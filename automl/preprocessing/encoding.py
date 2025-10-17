# encoding.py

# Standard library imports
from typing import Any, Optional

# Third-party imports
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# The function signature matches the one in column_handling.py and outliers.py
def auto_encode_features(
    X: DataFrame,
    y: Optional[Series],
    fit: bool,
    step_params: dict[str, Any],
    logger: Any,
    **config: Any,
) -> tuple[DataFrame, Optional[Series], dict[str, Any]]:
    """
    Applies appropriate encoding (One-Hot or Ordinal) to categorical features
    based on their cardinality.

    - Low cardinality (< OHE_CARDINALITY_THRESHOLD) features are One-Hot Encoded.
    - Medium cardinality features are Ordinal Encoded as a compact fallback.

    The function operates on the provided DataFrame X and stores/applies fitted
    encoders via step_params.
    """
    X_work = X.copy()
    if fit:
        # Configuration: Default threshold for switching from OHE to a compact encoder
        OHE_THRESHOLD = config.get("OHE_CARDINALITY_THRESHOLD", 10)

        # Identify all categorical features (object/string and category dtypes)
        categorical_cols = X_work.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if not categorical_cols:
            logger.info("No categorical features found for encoding.")
            return X, y, step_params

        logger.info(f"Categorical features to encode: {categorical_cols}")

        ohe_cols = []
        ordinal_cols = []

        # 1. Determine the appropriate encoder for each column
        for col in categorical_cols:
            cardinality = X_work[col].nunique()
            if cardinality <= OHE_THRESHOLD:
                ohe_cols.append(col)
            else:
                ordinal_cols.append(col)
        step_params["encoding"] = {}
        step_params["encoding"]["ohe_cols"] = ohe_cols
        step_params["encoding"]["ordinal_cols"] = ordinal_cols
        
    ohe_cols = step_params["encoding"]["ohe_cols"]
    ordinal_cols = step_params["encoding"]["ordinal_cols"]
    # --- 2. One-Hot Encoding (for low-cardinality nominal features) ---
    if ohe_cols:
        encoder_key = "ohe_encoder"

        if fit:
            logger.debug(
                f"Fitting OneHotEncoder on low-cardinality features: {ohe_cols}"
            )
            # handle_unknown='ignore': prevents errors on unseen categories in test data
            ohe = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore", dtype=np.float64
            )
            ohe.fit(X_work[ohe_cols])
            step_params[encoder_key] = {"encoder": ohe, "cols": ohe_cols}

        if encoder_key in step_params:
            ohe_data = step_params[encoder_key]
            ohe: OneHotEncoder = ohe_data["encoder"]
            cols_to_transform = ohe_data["cols"]

            logger.debug(
                f"Applying OneHotEncoder to features: {cols_to_transform}"
            )

            # Transform data and get new column names
            X_encoded_array = ohe.transform(X_work[cols_to_transform])
            feature_names = list(
                ohe.get_feature_names_out(input_features=cols_to_transform)
            )

            # Ensure we have a dense numpy array (handle sparse output for older encoders)
            if hasattr(X_encoded_array, "toarray"):
                X_encoded_array = X_encoded_array.toarray() # type: ignore
            else:
                X_encoded_array = np.asarray(X_encoded_array)

            # Create DataFrame and merge
            X_encoded_df = pd.DataFrame(
                X_encoded_array, columns=feature_names, index=X_work.index
            )
            # Drop original columns and concatenate new one-hot features
            X_work = pd.concat(
                [X_work.drop(columns=cols_to_transform), X_encoded_df], axis=1
            )

    # --- 3. Ordinal Encoding (for medium-cardinality features) ---
    if ordinal_cols:
        encoder_key = "ordinal_encoder"

        if fit:
            logger.debug(
                f"Fitting OrdinalEncoder on medium-cardinality features: {ordinal_cols}"
            )
            # unknown_value=-1 and handle_unknown='use_encoded_value':
            # Ensures unseen categories in test data are assigned a safe, distinct value (-1)
            ordinal = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int64,
            )
            ordinal.fit(X_work[ordinal_cols])
            step_params[encoder_key] = {
                "encoder": ordinal,
                "cols": ordinal_cols,
            }

        if encoder_key in step_params:
            ordinal_data = step_params[encoder_key]
            ordinal: OrdinalEncoder = ordinal_data["encoder"]
            cols_to_transform = ordinal_data["cols"]

            logger.debug(
                f"Applying OrdinalEncoder to features: {cols_to_transform}"
            )

            # Transform and replace the columns in place
            X_work[cols_to_transform] = ordinal.transform(
                X_work[cols_to_transform]
            )

    logger.info(f"Categorical encoding completed. New shape: {X_work.shape}")

    return X_work, y, step_params
