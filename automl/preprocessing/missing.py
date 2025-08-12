# internal imports
from library import Logger, infer_dtype, check_classification

# external imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import (
    OrdinalEncoder,
)
from time import time
from typing import Tuple, Optional, Dict, Any


def decide_knn_imputation(X, column: str) -> bool:
    """
    Decide if kNN imputation is suitable for the specified column of self.X_train.
    Returns True if conditions met, False otherwise.
    """

    zero_var_cols = []
    for col in X.columns:
        # Only consider numeric columns since variance for strings is not defined
        if pd.api.types.is_numeric_dtype(X[col]):
            # Drop NA before variance calc to avoid errors
            if X[col].dropna().nunique() <= 1:
                zero_var_cols.append(col)

    col_data = X[column]
    total_len = len(col_data)

    # 1. Check missingness proportion
    missing_ratio = col_data.isna().mean()
    if missing_ratio < 0.05 or missing_ratio > 0.30:
        # Missingness too low or too high
        return False

    # 2. Check dataset size
    if total_len < 200:
        return False

    # 3. Check for correlated features with absolute Pearson correlation > 0.3
    # Only consider rows without missing values in column and other features
    not_null_idx = col_data.dropna().index
    if len(not_null_idx) < 50:
        # Not enough complete data for correlation calculation
        return False

    X_no_missing = X.loc[not_null_idx].drop(columns=[column])
    target_no_missing = col_data.loc[not_null_idx]

    if X_no_missing.empty:
        # No other features to correlate with
        return False

    var_threshold = 1e-3  # something slightly below your lowest variance
    non_constant_cols = [
        col for col in X_no_missing if X_no_missing[col].var() > var_threshold
    ]
    corr_series = (
        X_no_missing[non_constant_cols].corrwith(target_no_missing).abs()
    )
    # Keep only numeric correlations (drop NaNs)
    corr_series = corr_series.dropna()

    if corr_series.empty or all(corr_series < 0.25):
        # No sufficiently correlated features
        return False

    # 4. Cross-validate imputation quality

    # Prepare data for imputation: We will simulate missingness on part of existing non-missing data
    col_non_missing = col_data.dropna()
    if len(col_non_missing) < 100:
        # Not enough data to simulate missingness robustly
        return False

    # Randomly mask 10% of known values in the column
    np.random.seed(42)
    mask_indices = np.random.choice(
        col_non_missing.index,
        size=int(0.1 * len(col_non_missing)),
        replace=False,
    )

    col_simulated = col_data.copy()
    col_simulated.loc[mask_indices] = np.nan

    # Create DataFrame for imputation (other features as is)
    X_for_impute = X.copy()
    X_for_impute[column] = col_simulated

    # Mean imputation baseline on the column only
    mean_imputer = SimpleImputer(strategy="mean")
    col_mean_imputed = mean_imputer.fit_transform(
        X_for_impute[[column]]
    ).ravel()

    # kNN imputation (neighbors=5 default)
    imputer = KNNImputer(n_neighbors=5)
    # kNN uses all columns, so impute full DataFrame
    start_time = time()
    X_knn_imputed = imputer.fit_transform(X_for_impute)
    elapsed_time = time() - start_time
    col_knn_imputed = X_knn_imputed[:, X_for_impute.columns.get_loc(column)]

    # Calculate RMSE of imputation only on masked indices (where we know true values)
    true_values = col_data.loc[mask_indices]

    # Get integer positional indices for proper NumPy indexing
    mask_pos = X_for_impute.index.get_indexer(mask_indices)

    mean_rmse = np.sqrt(
        mean_squared_error(true_values, col_mean_imputed[mask_pos])
    )
    knn_rmse = np.sqrt(
        mean_squared_error(true_values, col_knn_imputed[mask_pos])
    )
    # 5. Evaluate improvement and computational cost
    if knn_rmse < 0.9 * mean_rmse and elapsed_time < 10:
        return True
    else:
        return False


def get_feature_importances(X, y) -> dict[str, np.ndarray]:
    """
    Compute feature importances using a Random Forest model fit on encoded training data.

    Encodes categorical features using OrdinalEncoder, fits a classifier or regressor
    based on the target type, and returns feature importance scores.

    Returns:
        dict[str, np.ndarray]: Dictionary mapping feature names to importance scores.
    """
    X_train_enc = X.copy()

    # Identify categorical columns by dtype inference
    cat_cols = []
    for col in X_train_enc.columns:
        dtype = infer_dtype(X_train_enc[col])
        if dtype in {"string", "category", "boolean", "object"}:
            cat_cols.append(col)

    # Encode categorical features with OrdinalEncoder
    if cat_cols:
        enc = OrdinalEncoder()
        X_train_enc[cat_cols] = enc.fit_transform(X_train_enc[cat_cols])

    # Determine if target is classification or regression
    y_train = y

    is_regression = not check_classification(target=y_train)

    # Alternatively, you might check number of unique classes or task knowledge
    # Here a simple numeric dtype check is used

    # Fit appropriate model
    if is_regression:
        model = RandomForestRegressor(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train_enc, y_train)

    importances = dict(zip(X_train_enc.columns, model.feature_importances_))
    return importances


def handle_missing_values_cat(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit: bool,
    step_params: Dict[str, Any] = {},
    target_aware: bool = True,
    logger: Logger,
    categorical_only: bool = False,
    step_outputs: Dict[str, Any] = {},
    col_importance_thresh: float = 0.01,
    col_missing_thresh: float = 0.5,
    row_missing_thresh: float = 0.05,
    max_row_drop_frac: float = 0.1,
    skew_thresh: float = 0.5,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Handle missing values by selectively dropping columns or rows, and imputing values.

    - Drops columns with missing values above `col_missing_thresh` and low feature importance.
    - Drops rows with weighted missingness above `row_missing_thresh` if fraction dropped <= `max_row_drop_frac`.
    - Imputes missing values in numerical columns using mean or median imputation based on skewness.
    - Imputes missing values in categorical columns using most frequent values.

    Args:
        col_importance_thresh (float): Threshold for feature importance under which columns may be dropped if missing.
        col_missing_thresh (float): Fraction of missing values in a column above which to consider dropping the column.
        row_missing_thresh (float): Weighted fraction of missing values per row above which to consider dropping the row.
        max_row_drop_frac (float): Maximum allowable fraction of rows to drop in missing value handling.
        skew_thresh (float): Threshold for absolute skewness to switch between mean and median imputation in numeric columns.
    """

    # Helper function to apply imputers
    def apply_imputers(X, imputers):
        X = X.copy()
        for imp, cols in imputers.values():
            if cols:
                try:
                    if isinstance(imp, KNNImputer):
                        # Only replace knn_cols subset after full transform
                        # transform entire dataframe to impute knn_cols correctly
                        imputed_full = imp.transform(X)
                        for col in cols:
                            idx = X.columns.get_loc(col)
                            X[col] = imputed_full[:, idx]
                    else:
                        X[cols] = imp.transform(X[cols])
                except Exception as e:
                    logger.warning(
                        f"[YELLOW] - Imputer failed on columns {cols}: {e}"
                    )
        return X

    if fit:
        X_train = X

        logger.debug(
            msg=f"[GREEN]- Processing missing {"categorical" if categorical_only else "numeric"} values"
        )
        feature_importances = get_feature_importances(X, y)
        # Dropping columns (unchanged)
        missing_per_col = X_train.isna().mean()
        missing_per_col_dict = X_train.isna().sum().to_dict()
        cols_done = [
            col
            for col in missing_per_col_dict
            if missing_per_col_dict[col] == 0
        ]
        feature_importances = {
            k: v for k, v in feature_importances.items() if k in X_train.columns
        }
        total_imp = sum(feature_importances.values())
        norm_importances = (
            {k: 1 / len(feature_importances) for k in feature_importances}
            if total_imp == 0
            else {k: v / total_imp for k, v in feature_importances.items()}
        )

        cols_to_drop = [
            col
            for col in X_train.columns
            if missing_per_col[col] > col_missing_thresh
            and norm_importances.get(col, 0) < col_importance_thresh
        ]

        if cols_to_drop:
            for col in cols_to_drop:
                logger.debug(
                    f"[GREEN]  - Dropping {col} because of too much missing values"
                )
        X_train = X_train.drop(columns=cols_to_drop)
        step_params["cols_to_drop"] = cols_to_drop
        feature_importances = {
            k: v
            for k, v in feature_importances.items()
            if k not in cols_to_drop
        }

        # Dropping rows (unchanged)
        missing_mask = X_train.isna()
        weights = pd.DataFrame(0, index=X_train.index, columns=X_train.columns)
        for col in X_train.columns:
            weights[col] = missing_mask[col].astype(
                float
            ) * norm_importances.get(col, 0)
        weighted_missing_per_row = weights.sum(axis=1)
        rows_to_drop = weighted_missing_per_row[
            weighted_missing_per_row > row_missing_thresh
        ].index
        fraction_rows_to_drop = len(rows_to_drop) / len(X_train)
        if fraction_rows_to_drop <= max_row_drop_frac:
            X_train = X_train.drop(index=rows_to_drop)
            y = y.drop(index=rows_to_drop)
            logger.debug(
                f"[GREEN]  - Dropped {len(rows_to_drop)} rows because of missing values."
            )
        else:
            rows_to_drop = pd.Index([])
        if not categorical_only:
            # Step to decide kNN columns
            knn_cols = []
            for col in X_train.columns:
                if col not in cols_done:
                    if decide_knn_imputation(X=X, column=col):
                        knn_cols.append(col)
            if knn_cols:
                logger.debug(
                    f"[GREEN]  - Columns selected for KNN imputation: {knn_cols}"
                )
        else:
            knn_cols = []

        # Determine categorical and numerical columns
        cat_cols, num_cols = [], []
        for col in X_train.columns:
            if col not in cols_done:
                dtype_infer = infer_dtype(X_train[col])
                if dtype_infer in ["string", "category", "boolean", "object"]:
                    cat_cols.append(col)
                else:
                    num_cols.append(col)

        # Remove KNN columns from mean/median imputers since they will be handled separately
        cat_cols = [col for col in cat_cols if col not in knn_cols]
        # Prepare imputers dict to hold fitted imputers for each type
        imputers = {}
        if not categorical_only:
            num_cols = [col for col in num_cols if col not in knn_cols]
            # Skewness on numerical columns excluding kNN columns
            skewness = (
                X_train[num_cols].skew().abs()
                if num_cols
                else pd.Series(dtype=float)
            )

            # Numerical low skew mean imputer
            num_cols_low_skew = (
                skewness[skewness <= skew_thresh].index.tolist()
                if not skewness.empty
                else []
            )
            if num_cols_low_skew:
                imp_mean = SimpleImputer(strategy="mean")
                imp_mean.fit(X_train[num_cols_low_skew])
                imputers["num_mean"] = (imp_mean, num_cols_low_skew)
                for col in num_cols_low_skew:
                    logger.debug(
                        f"[GREEN]  - Imputing missing values with 'mean'(low skew numeric) for column {col}"
                    )

            # Numerical high skew median imputer
            num_cols_high_skew = (
                skewness[skewness > skew_thresh].index.tolist()
                if not skewness.empty
                else []
            )
            if num_cols_high_skew:
                imp_median = SimpleImputer(strategy="median")
                imp_median.fit(X_train[num_cols_high_skew])
                imputers["num_median"] = (imp_median, num_cols_high_skew)
                for col in num_cols_high_skew:
                    logger.debug(
                        f"[GREEN]  - Imputing missing values with 'median'(high skew numeric) for column {col}"
                    )

        # Categorical mode imputer
        if cat_cols:
            imp_mode = SimpleImputer(strategy="most_frequent")
            imp_mode.fit(X_train[cat_cols])
            imputers["cat_mode"] = (imp_mode, cat_cols)
            for col in cat_cols:
                logger.debug(
                    f"[GREEN]  - Imputing missing values with 'mode'(category) for column {col}"
                )

        # KNN imputer on knn_cols (if any)
        if not categorical_only and knn_cols:
            # KNN imputer requires all features as context; use all columns except dropped and rows dropped
            # Fit on full dataset columns (without dropped columns)
            knn_imputer = KNNImputer(n_neighbors=5)
            knn_imputer.fit(
                X_train
            )  # fit on all features for neighbor distances
            imputers["knn"] = (knn_imputer, knn_cols)
            for col in knn_cols:
                logger.debug(
                    f"[GREEN]  - Imputing missing values with 'KNN' for column {col}"
                )

        # Apply imputers to all splits
        X = apply_imputers(X_train, imputers)
        step_params["imputers"] = imputers
        return X, y, step_params
    else:
        if categorical_only:
            my_params = step_outputs["handle_missing_values_cat"]
        else:
            my_params = step_outputs["handle_missing_values_num"]
        cols_to_drop = my_params["cols_to_drop"]
        imputers = my_params["imputers"]
        X = X.drop(columns=cols_to_drop)
        X = apply_imputers(X, imputers)
        return X, y, my_params


def handle_missing_values_num(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit: bool,
    step_params: Dict[str, Any] = {},
    target_aware: bool = True,
    logger: Logger,
    categorical_only: bool = False,
    step_outputs: Dict[str, Any] = {},
    col_importance_thresh: float = 0.01,
    col_missing_thresh: float = 0.5,
    row_missing_thresh: float = 0.05,
    max_row_drop_frac: float = 0.1,
    skew_thresh: float = 0.5,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Handle missing values by selectively dropping columns or rows, and imputing values.

    - Drops columns with missing values above `col_missing_thresh` and low feature importance.
    - Drops rows with weighted missingness above `row_missing_thresh` if fraction dropped <= `max_row_drop_frac`.
    - Imputes missing values in numerical columns using mean or median imputation based on skewness.
    - Imputes missing values in categorical columns using most frequent values.

    Args:
        col_importance_thresh (float): Threshold for feature importance under which columns may be dropped if missing.
        col_missing_thresh (float): Fraction of missing values in a column above which to consider dropping the column.
        row_missing_thresh (float): Weighted fraction of missing values per row above which to consider dropping the row.
        max_row_drop_frac (float): Maximum allowable fraction of rows to drop in missing value handling.
        skew_thresh (float): Threshold for absolute skewness to switch between mean and median imputation in numeric columns.
    """

    # Helper function to apply imputers
    def apply_imputers(X, imputers):
        X = X.copy()
        for imp, cols in imputers.values():
            if cols:
                try:
                    if isinstance(imp, KNNImputer):
                        # Only replace knn_cols subset after full transform
                        # transform entire dataframe to impute knn_cols correctly
                        imputed_full = imp.transform(X)
                        for col in cols:
                            idx = X.columns.get_loc(col)
                            X[col] = imputed_full[:, idx]
                    else:
                        X[cols] = imp.transform(X[cols])
                except Exception as e:
                    logger.warning(
                        f"[YELLOW] - Imputer failed on columns {cols}: {e}"
                    )
        return X

    if fit:
        X_train = X

        logger.debug(
            msg=f"[GREEN]- Processing missing {"categorical" if categorical_only else "numeric"} values"
        )
        feature_importances = get_feature_importances(X, y)
        # Dropping columns (unchanged)
        missing_per_col = X_train.isna().mean()
        missing_per_col_dict = X_train.isna().sum().to_dict()
        cols_done = [
            col
            for col in missing_per_col_dict
            if missing_per_col_dict[col] == 0
        ]
        feature_importances = {
            k: v for k, v in feature_importances.items() if k in X_train.columns
        }
        total_imp = sum(feature_importances.values())
        norm_importances = (
            {k: 1 / len(feature_importances) for k in feature_importances}
            if total_imp == 0
            else {k: v / total_imp for k, v in feature_importances.items()}
        )

        cols_to_drop = [
            col
            for col in X_train.columns
            if missing_per_col[col] > col_missing_thresh
            and norm_importances.get(col, 0) < col_importance_thresh
        ]

        if cols_to_drop:
            for col in cols_to_drop:
                logger.debug(
                    f"[GREEN]  - Dropping {col} because of too much missing values"
                )
        X_train = X_train.drop(columns=cols_to_drop)
        step_params["cols_to_drop"] = cols_to_drop
        feature_importances = {
            k: v
            for k, v in feature_importances.items()
            if k not in cols_to_drop
        }

        # Dropping rows (unchanged)
        missing_mask = X_train.isna()
        weights = pd.DataFrame(0, index=X_train.index, columns=X_train.columns)
        for col in X_train.columns:
            weights[col] = missing_mask[col].astype(
                float
            ) * norm_importances.get(col, 0)
        weighted_missing_per_row = weights.sum(axis=1)
        rows_to_drop = weighted_missing_per_row[
            weighted_missing_per_row > row_missing_thresh
        ].index
        fraction_rows_to_drop = len(rows_to_drop) / len(X_train)
        if fraction_rows_to_drop <= max_row_drop_frac:
            X_train = X_train.drop(index=rows_to_drop)
            y = y.drop(index=rows_to_drop)
            logger.debug(
                f"[GREEN]  - Dropped {len(rows_to_drop)} rows because of missing values."
            )
        else:
            rows_to_drop = pd.Index([])
        if not categorical_only:
            # Step to decide kNN columns
            knn_cols = []
            for col in X_train.columns:
                if col not in cols_done:
                    if decide_knn_imputation(X=X, column=col):
                        knn_cols.append(col)
            if knn_cols:
                logger.debug(
                    f"[GREEN]  - Columns selected for KNN imputation: {knn_cols}"
                )
        else:
            knn_cols = []

        # Determine categorical and numerical columns
        cat_cols, num_cols = [], []
        for col in X_train.columns:
            if col not in cols_done:
                dtype_infer = infer_dtype(X_train[col])
                if dtype_infer in ["string", "category", "boolean", "object"]:
                    cat_cols.append(col)
                else:
                    num_cols.append(col)

        # Remove KNN columns from mean/median imputers since they will be handled separately
        cat_cols = [col for col in cat_cols if col not in knn_cols]
        # Prepare imputers dict to hold fitted imputers for each type
        imputers = {}
        if not categorical_only:
            num_cols = [col for col in num_cols if col not in knn_cols]
            # Skewness on numerical columns excluding kNN columns
            skewness = (
                X_train[num_cols].skew().abs()
                if num_cols
                else pd.Series(dtype=float)
            )

            # Numerical low skew mean imputer
            num_cols_low_skew = (
                skewness[skewness <= skew_thresh].index.tolist()
                if not skewness.empty
                else []
            )
            if num_cols_low_skew:
                imp_mean = SimpleImputer(strategy="mean")
                imp_mean.fit(X_train[num_cols_low_skew])
                imputers["num_mean"] = (imp_mean, num_cols_low_skew)
                for col in num_cols_low_skew:
                    logger.debug(
                        f"[GREEN]  - Imputing missing values with 'mean'(low skew numeric) for column {col}"
                    )

            # Numerical high skew median imputer
            num_cols_high_skew = (
                skewness[skewness > skew_thresh].index.tolist()
                if not skewness.empty
                else []
            )
            if num_cols_high_skew:
                imp_median = SimpleImputer(strategy="median")
                imp_median.fit(X_train[num_cols_high_skew])
                imputers["num_median"] = (imp_median, num_cols_high_skew)
                for col in num_cols_high_skew:
                    logger.debug(
                        f"[GREEN]  - Imputing missing values with 'median'(high skew numeric) for column {col}"
                    )

        # Categorical mode imputer
        if cat_cols:
            imp_mode = SimpleImputer(strategy="most_frequent")
            imp_mode.fit(X_train[cat_cols])
            imputers["cat_mode"] = (imp_mode, cat_cols)
            for col in cat_cols:
                logger.debug(
                    f"[GREEN]  - Imputing missing values with 'mode'(category) for column {col}"
                )

        # KNN imputer on knn_cols (if any)
        if not categorical_only and knn_cols:
            # KNN imputer requires all features as context; use all columns except dropped and rows dropped
            # Fit on full dataset columns (without dropped columns)
            knn_imputer = KNNImputer(n_neighbors=5)
            knn_imputer.fit(
                X_train
            )  # fit on all features for neighbor distances
            imputers["knn"] = (knn_imputer, knn_cols)
            for col in knn_cols:
                logger.debug(
                    f"[GREEN]  - Imputing missing values with 'KNN' for column {col}"
                )

        # Apply imputers to all splits
        X = apply_imputers(X_train, imputers)
        step_params["imputers"] = imputers
        return X, y, step_params
    else:
        if categorical_only:
            my_params = step_outputs["handle_missing_values_cat"]
        else:
            my_params = step_outputs["handle_missing_values_num"]
        cols_to_drop = my_params["cols_to_drop"]
        imputers = my_params["imputers"]
        X = X.drop(columns=cols_to_drop)
        X = apply_imputers(X, imputers)
        return X, y, my_params
