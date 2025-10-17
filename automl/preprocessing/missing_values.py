# Standard library imports
from dataclasses import dataclass
from typing import Any, Optional,  cast

# Third-party imports
import pandas as pd
from pandas import DataFrame, Series
from automl.library import Logger
from sklearn.impute import IterativeImputer

@dataclass(slots=True)
class ImputationInfo:
    """
    Stores the decided imputation method and value for a single column.
    """

    method: str
    fill_value: Optional[Any] = None


def decide_imputation_strategy(
    X_train: DataFrame,
    logger: Any,
) -> dict[str, ImputationInfo]:
    """
    Analyzes the DataFrame and determines the optimal imputation strategy for each column.

    The decision process is filtered by the `categorical_only` flag.

    Args:
        X_train: The training DataFrame with potentially missing values.
        categorical_only: If True, only consider object/category columns.
        logger: The custom logger instance.

    Returns:
        A dictionary mapping each relevant column name with missing values to its ImputationInfo.
    """

    # --- 1. Global Decision: MICE vs. Simple Imputation (Applicable only to the full X_train) ---
    MICE_SAMPLES_THRESHOLD = 50_000
    MICE_FEATURES_THRESHOLD = 100
    MICE_CELLS_THRESHOLD = 1_000_000

    n_samples = len(X_train)
    n_features = X_train.shape[1]
    n_missing_cells = X_train.isnull().sum().sum()

    use_mice = (
        n_samples <= MICE_SAMPLES_THRESHOLD
        and n_features <= MICE_FEATURES_THRESHOLD
        and n_missing_cells <= MICE_CELLS_THRESHOLD
        and n_missing_cells > 0
    )

    imputation_strategies: dict[str, ImputationInfo] = {}
    missing_cols_series: Series = X_train.isnull().any()

    # Filter columns to only include those that match the type criteria AND have missing values
    for col, has_missing in missing_cols_series.items():
        if not has_missing:
            continue

        series = X_train[str(col)]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # --- 2. Assign Strategy Based on Global Decision and Column Type ---

        if use_mice:
            imputation_strategies[str(col)] = ImputationInfo(method="MICE")
        else:
            # Fallback to simple per-column imputation
            if is_numeric:
                # Decide between mean and median based on skewness
                # Exclude NaNs for accurate skewness calculation
                non_missing_data = series.dropna()

                # Check for zero standard deviation (constant after dropping NaNs)
                # If all non-missing values are the same, use the value itself
                if non_missing_data.std() == 0:
                    fill_val = (
                        non_missing_data.iloc[0]
                        if not non_missing_data.empty
                        else 0
                    )
                    imputation_strategies[str(col)] = ImputationInfo(
                        method="constant", fill_value=fill_val
                    )
                    logger.debug(
                        f"Col '{col}': Constant data. Imputing with constant value."
                    )
                    continue

                # Use Fisher-Pearson coefficient of skewness
                try:
                    skewness = cast(float, non_missing_data.skew())
                except Exception:
                    # Fallback for columns where skewness fails (e.g., single unique value, but with NaNs)
                    skewness = 0.0

                SKEW_THRESHOLD = 1.0  # Threshold for significant skew

                if abs(skewness) > SKEW_THRESHOLD:
                    fill_val = series.median()
                    imputation_strategies[str(col)] = ImputationInfo(
                        method="median", fill_value=fill_val
                    )
                    logger.debug(
                        f"Col '{col}': Skewness={skewness:.2f}. Using median."
                    )
                else:
                    fill_val = series.mean()
                    imputation_strategies[str(col)] = ImputationInfo(
                        method="mean", fill_value=fill_val
                    )
                    logger.debug(
                        f"Col '{col}': Skewness={skewness:.2f}. Using mean."
                    )
            else:
                # For categorical/object columns, always use the mode
                # The .mode() method returns a Series, so we take the first value.
                # If mode returns multiple values (multimodal), pandas usually returns the smallest/first.
                fill_val = series.mode()[0]
                imputation_strategies[str(col)] = ImputationInfo(
                    method="mode", fill_value=fill_val
                )
                logger.debug(
                    f"Col '{col}': Categorical. Using mode='{fill_val}'."
                )

    if not imputation_strategies:
        logger.info(
            "No missing values found in columns."
        )

    return imputation_strategies


def handle_missing_values(
    X: DataFrame,
    y: Optional[Series],
    fit: bool,
    step_params: dict[str, Any],
    categorical_only: bool,
    logger: Logger,
    **config: Any,
) -> tuple[DataFrame, Optional[Series], dict[str, Any]]:
    """
    Handles missing values in the DataFrame X by applying pre-determined
    simple imputation (mean/median/mode) or fitting/applying MICE.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix.
    y : Optional[pd.Series]
        The target variable (passed through).
    fit : bool
        If True, the function is in the fitting/training phase (fit and transform).
    step_params : dict[str, Any]
        A dictionary to store/retrieve learned parameters (strategies, bounds, etc.).
    categorical_only: bool
        If True, only apply imputation to categorical/object columns (simple imputation).
        If False, apply to numeric columns (simple or MICE imputation).
    logger : Logger
        The logging utility object.
    meta_data : dict[str, dict[str, Any]]
        Metadata about the dataset and features (passed through).
    **config : Any
        Global configuration settings, which may contain parameters like 'random_state'.

    Returns
    -------
    tuple[pd.DataFrame, Optional[pd.Series], dict[str, Any]]
        The transformed feature matrix X, the original target y, and the updated step_params.
    """
    X_work = X.copy()
    imputation_map = step_params["general_info"].get("missing_imputation", {})

    # ----------------------------------------------------------------------
    # 1. Simple Imputation (Mean/Median/Mode) - Applied column-wise
    # This handles all categorical imputation ('mode') and simple numeric imputation.
    # ----------------------------------------------------------------------
    for col in X_work.columns:
        is_numeric = pd.api.types.is_numeric_dtype(X_work[col])

        # Check if column type matches the requirement (numeric if not categorical_only, else non-numeric)
        if is_numeric != categorical_only:

            if col in imputation_map:
                # Assuming imputation_map[col] is an ImputationInfo object
                imputation_info = imputation_map[col]
                method = imputation_info.method
                fill_value = imputation_info.fill_value

                # Apply simple imputation if the method is not MICE and a fill value exists
                if method != "MICE" and fill_value is not None:
                    # Apply the imputation using the stored value
                    X_work[col] = X_work[col].fillna(fill_value)

                    # Logging
                    log_prefix = "FITTED" if fit else "APPLIED"
                    log_type = "Numeric" if is_numeric else "Categorical"

                    logger.info(
                        f"Missing values {log_prefix} (Simple) for column '{col}' ({log_type}). "
                        f"Method: '{method}'. Fill Value: '{fill_value}'."
                    )

    # ----------------------------------------------------------------------
    # 2. MICE Imputation (IterativeImputer) - Applied only to numeric columns
    # MICE must run on a collection of features simultaneously.
    # ----------------------------------------------------------------------
    if not categorical_only:

        # Identify numeric columns designated for MICE that still have NaNs
        mice_columns = [
            col
            for col in X_work.columns
            if pd.api.types.is_numeric_dtype(X_work[col])
            and col in imputation_map
            and imputation_map[col].method == "MICE"
            and X_work[col].isnull().any()
        ]

        if mice_columns:

            # Create a working copy of the MICE columns for imputation
            X_mice = X_work[mice_columns].copy()

            if fit:
                logger.info(
                    f"MICE: Fitting IterativeImputer on {len(mice_columns)} columns: {mice_columns}"
                )

                # Instantiate and fit the imputer
                imputer = IterativeImputer(
                    random_state=config.get("random_state", 42),
                    max_iter=config.get("mice_max_iter", 10),
                )

                # Fitting also transforms the training data
                X_mice_imputed = imputer.fit_transform(X_mice)

                # Store the fitted imputer and the list of columns it was trained on
                step_params["mice_imputer"] = imputer
                step_params["mice_columns"] = mice_columns

                logger.info("MICE: IterativeImputer fitted and stored.")

            else:  # fit=False (Transform mode)

                # Check if the imputer was fitted during the training phase
                if "mice_imputer" not in step_params:
                    logger.warning(
                        "MICE: IterativeImputer was not found in step_params. Skipping MICE imputation for transform data."
                    )
                    return X_work, y, step_params

                imputer = step_params["mice_imputer"]
                # Use the column list from the fit stage to ensure consistency
                mice_columns = step_params["mice_columns"]

                logger.info(
                    f"MICE: Applying fitted IterativeImputer to {len(mice_columns)} columns."
                )

                # Apply the transformation
                X_mice_imputed = imputer.transform(X_mice)

            # Update the working DataFrame with the imputed MICE values
            # The output of IterativeImputer is a NumPy array, requiring explicit type conversion back.
            X_work.loc[:, mice_columns] = X_mice_imputed

            # Attempt to restore original data types to comply with the project requirement
            for col in mice_columns:
                original_dtype = X[col].dtype
                # If the original was an integer type (including nullable Int64), attempt to cast back
                if pd.api.types.is_integer_dtype(original_dtype) or str(
                    original_dtype
                ).startswith("Int"):
                    try:
                        # Attempt to cast back to original type (e.g., Int64)
                        # This conversion will handle cases where the float imputation
                        # results in a whole number and can be safely converted to integer.
                        X_work[col] = X_work[col].astype(original_dtype)
                    except (TypeError, ValueError):
                        # If the MICE imputation resulted in fractional values, keep the float
                        logger.warning(
                            f"MICE: Column '{col}' could not be restored to original dtype {original_dtype} "
                            f"due to non-integer imputed values. Keeping as float."
                        )

    return X_work, y, step_params
