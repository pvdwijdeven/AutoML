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
import time
from typing import Tuple, Optional, Dict, Any
from pandas.api.types import is_numeric_dtype


def decide_knn_imputation(X: pd.DataFrame, column: str, logger: Logger) -> bool:
    """
    Decide if k-Nearest Neighbors (kNN) imputation is suitable for the specified column.

    The decision is based on multiple conditions:
    1. Missingness proportion in the column must be between 5% and 30%.
    2. Total dataset size must be at least 200 samples.
    3. There must be at least 50 complete cases (rows without missing values in the specified column).
    4. At least one other feature must have absolute Pearson correlation > 0.25 with the column.
    5. There must be sufficient data (at least 100 non-missing samples) to simulate missingness and evaluate imputation quality.
    6. kNN imputation must significantly improve RMSE over mean imputation (less than 90% of mean imputation RMSE) and complete within reasonable time (< 10 seconds).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing the data.
    column : str
        Name of the column to evaluate for kNN imputation suitability.

    Returns
    -------
    bool
        True if kNN imputation is suitable for the column, False otherwise.
    """
    zero_var_cols = []
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            if X[col].dropna().nunique() <= 1:
                zero_var_cols.append(col)
    col_data = X[column]
    total_len = len(col_data)

    # 1. Missingness proportion check
    missing_ratio = col_data.isna().mean()
    if missing_ratio < 0.05 or missing_ratio > 0.30:
        return False

    # 2. Dataset size check
    if total_len < 200:
        return False

    # 3. Check for correlated features
    not_null_idx = col_data.dropna().index
    if len(not_null_idx) < 50:
        return False

    X_no_missing = X.loc[not_null_idx].drop(columns=[column])
    target_no_missing = col_data.loc[not_null_idx]
    if X_no_missing.empty:
        return False

    var_threshold = 1e-3
    non_constant_cols = [
        col for col in X_no_missing if X_no_missing[col].var() > var_threshold  # type: ignore
    ]
    corr_series = (
        X_no_missing[non_constant_cols]
        .corrwith(target_no_missing)
        .abs()
        .dropna()
    )
    if corr_series.empty or all(corr_series < 0.25):
        return False

    # 4. Cross-validate imputation quality by simulating missingness
    col_non_missing = col_data.dropna()
    if len(col_non_missing) < 100:
        return False

    # Randomly mask 10% of non-missing values
    np.random.seed(42)
    mask_indices = np.random.choice(
        col_non_missing.index,
        size=int(0.1 * len(col_non_missing)),
        replace=False,
    )

    col_simulated = col_data.copy()
    col_simulated.loc[mask_indices] = np.nan
    X_for_impute = X.copy()
    X_for_impute[column] = col_simulated

    # Mean imputation baseline
    mean_imputer = SimpleImputer(strategy="mean")
    col_mean_imputed = mean_imputer.fit_transform(
        X_for_impute[[column]]
    ).ravel()

    # kNN imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    start_time = time.time()
    X_knn_imputed = knn_imputer.fit_transform(X_for_impute)
    elapsed_time = time.time() - start_time

    col_knn_imputed = X_knn_imputed[:, X_for_impute.columns.get_loc(column)]
    true_values = col_data.loc[mask_indices]
    mask_pos = X_for_impute.index.get_indexer(mask_indices)

    mean_rmse = np.sqrt(
        mean_squared_error(true_values, col_mean_imputed[mask_pos])
    )
    knn_rmse = np.sqrt(
        mean_squared_error(true_values, col_knn_imputed[mask_pos])
    )
    return knn_rmse < 0.9 * mean_rmse and elapsed_time < 10


def get_feature_importances(
    X: pd.DataFrame, y: pd.Series
) -> Dict[str, np.ndarray]:
    """
    Compute feature importances using a Random Forest model fitted on encoded features.

    Categorical features are ordinally encoded before fitting. The model type (classifier or regressor)
    is chosen based on the target type.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping feature names to their importance scores.
    """
    X_train_enc = X.copy()

    # Identify categorical columns by dtype inference
    cat_cols = [
        col
        for col in X_train_enc.columns
        if infer_dtype(X_train_enc[col])
        in {"string", "category", "boolean", "object"}
    ]

    # Encode categorical features with OrdinalEncoder
    if cat_cols:
        enc = OrdinalEncoder()
        X_train_enc[cat_cols] = enc.fit_transform(X_train_enc[cat_cols])

    # Determine if target is regression or classification
    is_regression = not check_classification(target=y)

    # Instantiate appropriate model
    if is_regression:
        model = RandomForestRegressor(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train_enc, y)

    importances = dict(zip(X_train_enc.columns, model.feature_importances_))
    return importances


def handle_missing_values(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    *,
    fit: bool,
    step_params: Dict[str, Any],
    logger: Logger,
    categorical_only: bool = False,
    meta_data: Dict[str, Any],
    col_importance_thresh: float = 0.01,
    col_missing_thresh: float = 0.5,
    skew_thresh: float = 0.5,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]:
    """
    Handle missing values by dropping columns or rows selectively, and imputing missing entries.

    Behavior during fit:
    - Identifies columns with missingness above `col_missing_thresh` and low importance below `col_importance_thresh`; marks these for dropping.
    - Optionally drops rows with weighted missingness > `row_missing_thresh` if the fraction dropped remains <= `max_row_drop_frac`.
    - Selects numeric columns suitable for kNN imputation.
    - Decides mean vs median imputation for numeric columns based on skewness threshold.
    - Categorical missing values imputed with most frequent values.
    - Fits all required imputers but does NOT transform data in fit mode.
    - Stores fitted imputers and drop lists in `step_params`.

    Behavior during transform:
    - Drops columns marked during fit.
    - Applies fitted imputers to the input data.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature DataFrame.
    y : Optional[pd.Series]
        Target variable (used for feature importance calculation).
    fit : bool
        Fit mode (True) or transform mode (False).
    step_params : Dict[str, Any]
        Dictionary to persist fitted imputers, columns to drop, etc.
    logger : Logger
        Logger instance for debug and warning messages.
    categorical_only : bool, default=False
        If True, only categorical features are processed.
    meta_data : Dict[str, Any]
        Metadata dictionary carrying precomputed parameters during pipeline runs.
    col_importance_thresh : float, default=0.01
        Threshold below which low-importance columns with high missingness will be dropped.
    col_missing_thresh : float, default=0.5
        Threshold fraction of missing values in a column to consider dropping it.
    skew_thresh : float, default=0.5
        Absolute skewness threshold to choose between mean and median imputation for numeric columns.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series], Optional[Dict[str, Any]]]
        The DataFrame with missing value handling applied or parameters updated,
        unchanged target, and updated step_params storing fitted imputers and drop info.

    Notes
    -----
    - In fit mode, data is NOT transformed; only imputers are fitted.
    - In the current implementation, row dropping logic placeholder is present but not used.
    """

    def apply_imputers(
        X_input: pd.DataFrame, imputers: Dict[str, Tuple[Any, list]]
    ) -> pd.DataFrame:
        X_copy = X_input.copy()
        for imp, cols in imputers.values():
            if cols:
                try:
                    if isinstance(imp, KNNImputer):
                        # kNN imputer expects full DataFrame to impute neighbors correctly
                        imputed_full = imp.transform(X_copy)
                        for col_name in cols:
                            idx = X_copy.columns.get_loc(col_name)
                            X_copy[col_name] = imputed_full[:, idx]
                    else:
                        X_copy[cols] = imp.transform(X_copy[cols])
                except Exception as e:
                    logger.warning(
                        f"[YELLOW] - Imputer failed on columns {cols}: {e}"
                    )
        return X_copy

    if fit:
        X_train = X.copy()
        logger.debug(
            f"[GREEN]- Processing missing {'categorical' if categorical_only else 'numeric'} values"
        )

        feature_importances = (
            get_feature_importances(X_train, y) if y is not None else {}
        )
        logger.debug(f"[GREEN]- feature importances: {feature_importances}")
        missing_per_col = X_train.isna().mean()
        missing_counts = X_train.isna().sum().to_dict()
        cols_done = [
            col for col, miss_count in missing_counts.items() if miss_count == 0
        ]

        # Filter feature importances to existing columns
        feature_importances = {
            k: v for k, v in feature_importances.items() if k in X_train.columns
        }
        total_importance = sum(feature_importances.values())
        norm_importances = (
            {k: 1 / len(feature_importances) for k in feature_importances}
            if total_importance == 0
            else {
                k: v / total_importance for k, v in feature_importances.items()
            }
        )

        # Identify columns to drop due to high missingness and low importance
        cols_to_drop = [
            col
            for col in X_train.columns
            if missing_per_col[col] > col_missing_thresh
            and norm_importances.get(col, 0) < col_importance_thresh
        ]
        for col in cols_to_drop:
            logger.debug(
                f"[GREEN]  - Dropping {col} because of too much missing values"
            )

        step_params["cols_to_drop"] = cols_to_drop
        # Update importances excluding dropped cols
        feature_importances = {
            k: v
            for k, v in feature_importances.items()
            if k not in cols_to_drop
        }

        # Decide which columns to consider for kNN imputation
        knn_cols = []
        if not categorical_only:
            for col in X_train.columns[X_train.isna().any()].tolist():
                logger.debug(f"{col}")
                if col not in cols_done and decide_knn_imputation(
                    X=X_train, column=col, logger=logger
                ):
                    knn_cols.append(col)
        logger.debug(
            f"[GREEN]  - Columns selected for KNN imputation: {knn_cols}"
        )
        # Define categorical and numeric columns excluding dropped ones
        cat_cols, num_cols = [], []
        for col in X_train.columns:
            dtype = infer_dtype(X_train[col])
            if dtype in ["string", "category", "boolean", "object"]:
                cat_cols.append(col)
            else:
                num_cols.append(col)

        # Exclude knn_cols from mean/median imputers
        cat_cols = [c for c in cat_cols if c not in knn_cols]
        num_cols = [c for c in num_cols if c not in knn_cols]

        imputers: Dict[str, Tuple[Any, list]] = {}

        if not categorical_only:
            # Skewness calculation on numeric cols excluding kNN
            skewness = (
                X_train[num_cols].skew().abs()
                if num_cols
                else pd.Series(dtype=float)
            )

            # Mean imputer for low skew columns
            low_skew_cols = (
                skewness[skewness <= skew_thresh].index.tolist()
                if not skewness.empty
                else []
            )
            if low_skew_cols:
                imp_mean = SimpleImputer(strategy="mean")
                imp_mean.fit(X_train[low_skew_cols])
                imputers["num_mean"] = (imp_mean, low_skew_cols)
                for col in low_skew_cols:
                    logger.debug(
                        f"[GREEN]  - Imputing missing values with 'mean' (low skew numeric) for column {col}"
                    )

            # Median imputer for high skew columns
            high_skew_cols = (
                skewness[skewness > skew_thresh].index.tolist()
                if not skewness.empty
                else []
            )
            if high_skew_cols:
                imp_median = SimpleImputer(strategy="median")
                imp_median.fit(X_train[high_skew_cols])
                imputers["num_median"] = (imp_median, high_skew_cols)
                for col in high_skew_cols:
                    logger.debug(
                        f"[GREEN]  - Imputing missing values with 'median' (high skew numeric) for column {col}"
                    )

        # Categorical mode imputer
        if cat_cols:
            imp_mode = SimpleImputer(strategy="most_frequent")
            imp_mode.fit(X_train[cat_cols])
            imputers["cat_mode"] = (imp_mode, cat_cols)
            for col in cat_cols:
                logger.debug(
                    f"[GREEN]  - Imputing missing values with 'mode' (category) for column {col}"
                )

        # kNN imputer for knn_cols
        if not categorical_only and knn_cols:
            knn_imputer = KNNImputer(n_neighbors=5)
            knn_imputer.fit(X_train)  # Fit on full dataset
            imputers["knn"] = (knn_imputer, knn_cols)
            for col in knn_cols:
                logger.debug(
                    f"[GREEN]  - Imputing missing values with 'KNN' for column {col}"
                )

        # Fit mode: do not transform X here, only prepare imputers
        logger.debug(f"[GREEN]  {imputers}")
        step_params["imputers"] = imputers
        return X, y, step_params

    else:  # transform mode
        # Use meta_data keys to find params; categorical only handled separately
        if categorical_only:
            my_params = meta_data.get("handle_missing_values_cat", {})
        else:
            my_params = meta_data.get("handle_missing_values_num", {})

        cols_to_drop = my_params.get("cols_to_drop", [])
        imputers = my_params.get("imputers", {})

        X_transformed = X.drop(columns=cols_to_drop, errors="ignore")
        X_transformed = apply_imputers(X_transformed, imputers)
        return X_transformed, y, my_params
