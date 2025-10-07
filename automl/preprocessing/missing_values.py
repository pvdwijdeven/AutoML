# Standard library imports
from dataclasses import dataclass
from typing import Any, Optional,  cast

# Third-party imports
import pandas as pd
from pandas import DataFrame, Series


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
