# standardizing.py

# Standard library imports
from typing import Any, Optional

# Third-party imports
import numpy as np
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler


def normalize_columns(
    X: DataFrame,
    y: Optional[Series],
    fit: bool,
    step_params: dict[str, Any],
    logger: Any,
    **config: Any,
) -> tuple[DataFrame, Optional[Series], dict[str, Any]]:
    """
    Applies Standard Scaling to numerical columns, explicitly excluding encoded features
    by checking the column lists stored in step_params.

    The function determines which columns to scale during fit=True and stores the
    Scaler object and the list of columns in step_params.
    """
    X_work = X.copy()
    scaler_key = "standard_scaler"

    # 1. Identify columns to EXCLUDE from scaling
    # Use .get() defensively in case the encoding step was skipped or failed
    encoded_cols_ordinal = (
        step_params.get("encoding", {})
        .get("ordinal_encoder", {})
        .get("cols", [])
    )

    # The new OHE columns are the features names created during transformation,
    # which are NOT stored in the original 'ohe_encoder' column list.
    # We must assume all columns that are NOT the original categorical features
    # and are NOT the original numerical features are the new OHE features.

    # A safer approach for OHE is to check the keys of the OneHotEncoder object,
    # but since the full set of final OHE column names is not easily tracked
    # via the original step_params structure, we use the exclusion method below
    # to target *non-encoded* numerical features.

    # Let's collect ALL numerical columns and filter:

    numeric_cols = X_work.select_dtypes(include=np.number).columns.tolist()

    # 2. Determine the columns to scale (only needed during FIT)
    if fit:

        # NOTE: This logic must assume that the final set of columns in X_work
        # that are numerical and were *not* originally numerical must be the OHE features.

        # Simplest exclusion rule: Scale all current numerical columns EXCEPT
        # those that are binary (0, 1) or look like ordinal integers (-1, 0, 1, 2...).
        # However, to be robust, we rely on the final column names.

        # 💡 Robust method: Check if the column name exists in the original categorical lists
        # or if it is a new OHE feature (which has a name like 'col_category').

        cols_to_scale = []

        for col in numeric_cols:
            # Check if this column is one of the original Ordinal Encoded columns
            if col in encoded_cols_ordinal:
                logger.debug(
                    f"Col '{col}' skipped normalization (Ordinal Encoded)."
                )
                continue

            # Check if this column resulted from One-Hot Encoding (e.g., 'Gender_Female')
            # This is complex without the full OHE names. We rely on a robust heuristic:
            # If a numerical column has only 0s and 1s, it's likely a OHE feature.
            # If it has few unique values (<= OHE_CARDINALITY_THRESHOLD used earlier),
            # it might be a OHE/OE column that shouldn't be scaled.

            is_ohe_like = X_work[col].isin([0, 1]).all()
            is_ordinal_like = (
                X_work[col].nunique()
                < config.get("OHE_CARDINALITY_THRESHOLD", 10) + 1
            )  # +1 for unknown value

            if is_ohe_like or is_ordinal_like:
                logger.debug(
                    f"Col '{col}' skipped normalization (OHE/Ordinal heuristic)."
                )
                continue

            cols_to_scale.append(col)

        # Store the list of columns found suitable for scaling
        step_params[scaler_key] = {"cols_to_scale": cols_to_scale}

    # 3. Load the column list regardless of fit status
    cols_to_transform = step_params.get(scaler_key, {}).get("cols_to_scale", [])

    if not cols_to_transform:
        logger.info("No continuous numerical features found for scaling.")
        return X, y, step_params

    # 4. Fit or load the scaler
    if fit:
        logger.debug(f"Fitting StandardScaler on features: {cols_to_transform}")
        scaler = StandardScaler()
        # Fit only on the selected columns
        scaler.fit(X_work[cols_to_transform])
        step_params[scaler_key]["scaler"] = scaler

    # 5. Transform the data
    if scaler_key in step_params and "scaler" in step_params[scaler_key]:
        scaler: StandardScaler = step_params[scaler_key]["scaler"]

        logger.debug(
            f"Applying StandardScaler to features: {cols_to_transform}"
        )

        # Transform the columns
        X_work[cols_to_transform] = scaler.transform(X_work[cols_to_transform])
    else:
        logger.warning(
            "Scaler not fitted, skipping scaling in transform mode."
        )

    logger.info("Normalization completed.")

    return X_work, y, step_params
