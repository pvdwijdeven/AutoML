from library import (
    Logger,
)
from eda import AutoML_EDA
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
import numpy as np
from library import infer_dtype, check_classification
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from pandas.api.types import is_numeric_dtype


class AutoML_Preprocess:
    def __init__(
        self,
        report_file: str,
        file_train: str,
        file_test: str = "",
        title: str = "",
        target: str = "",
        description: str = "",
        nogui=True,
        update_script: str = "",
        logger: Logger | None = None,
    ) -> None:

        self.report_file = report_file
        self.file_train = file_train
        self.file_test = file_test
        self.title = title
        self.target = target
        self.description = description
        self.nogui = nogui
        self.update_script = update_script
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger = logger
        self.eda = AutoML_EDA(
            report_file=self.report_file,
            file_train=self.file_train,
            file_test=self.file_test,
            title=self.title,
            target=self.target,
            description=self.description,
            nogui=self.nogui,
            update_script=self.update_script,
            logger=self.logger,
        )

    def split_train_test_val(self):
        assert self.eda.df_train is not None

        y = self.eda.df_train[self.target]
        X = self.eda.df_train.drop(columns=[self.target])

        # Determine if target is classification or regression
        # Simple heuristic: if target is numeric with many unique values, treat as regression

        is_classification = check_classification(y)

        if is_classification:
            stratify_obj = y
        else:
            # Regression case: create quantile bins for stratification
            try:
                # Use qcut for 10 bins, drop duplicates to avoid errors in small datasets
                stratify_obj = pd.qcut(y, q=10, duplicates="drop")
            except ValueError:
                # fallback: no stratification if binning fails
                stratify_obj = None

        # First split: train and temp (validation + test)
        if stratify_obj is not None:
            self.X_train, X_temp, self.y_train, y_temp = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify_obj,
            )
        else:
            self.X_train, X_temp, self.y_train, y_temp = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
            )

        # For validation/test split, apply same stratify logic on y_temp
        if stratify_obj is not None:
            if is_classification:
                stratify_temp = y_temp
            else:
                try:
                    stratify_temp = pd.qcut(y_temp, q=5, duplicates="drop")
                except ValueError:
                    stratify_temp = None
        else:
            stratify_temp = None

        if stratify_temp is not None:
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=0.5,
                random_state=42,
                stratify=stratify_temp,
            )
        else:
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )

        self.df_test = self.eda.df_test
        self.logger.info(
            f"[GREEN]- Split train/test/validation 80%/10%/10% ({self.X_train.shape[0]}/{self.X_test.shape[0]}/{self.X_val.shape[0]} rows)"
        )

    def drop_duplicate_rows(self):
        assert self.eda.df_train is not None
        num_rows_before = self.eda.df_train.shape[0]

        # Identify duplicate rows (all except the first occurrence)
        duplicate_mask = self.eda.df_train.duplicated(
            keep="first"
        )  # True for duplicates
        # Get index labels of duplicate rows that will be dropped
        dropped_indices = self.eda.df_train.index[duplicate_mask].tolist()
        # Drop duplicate rows
        self.eda.df_train = self.eda.df_train.drop_duplicates()
        num_rows_after = self.eda.df_train.shape[0]
        if num_rows_after < num_rows_before:
            self.logger.info(
                f"[GREEN]- Duplicate rows to be dropped: {num_rows_before - num_rows_after}"
            )
            self.logger.debug(
                f"[GREEN]  Dropped rows with indices: {dropped_indices}"
            )
        else:
            self.logger.info("[GREEN]- Duplicate rows to be dropped: None")

    def drop_duplicate_columns(self):
        # Step 1: Compute hash sum per column to detect duplicates
        hashes = self.X_train.apply(
            lambda col: pd.util.hash_pandas_object(col, index=False).sum()
        )

        # Step 2: Identify duplicated hashes (i.e., duplicate columns)
        duplicated_mask = hashes.duplicated()
        duplicate_columns = self.X_train.columns[duplicated_mask]

        # Log the duplicate columns (if any)
        self.logger.info(
            f"[GREEN]- Duplicate columns to be dropped: {list(duplicate_columns) if len(duplicate_columns) > 0 else 'None'}"
        )

        # Step 3: Drop duplicate columns from train, test, and val sets
        self.X_train = self.X_train.drop(columns=duplicate_columns)
        self.X_test = self.X_test.drop(
            columns=duplicate_columns, errors="ignore"
        )
        self.X_val = self.X_val.drop(columns=duplicate_columns, errors="ignore")
        if self.df_test is not None:
            self.df_test = self.df_test.drop(
                columns=duplicate_columns, errors="ignore"
            )

    def drop_constant_columns(self):
        # Vectorized approach: find columns where number of unique values is 1
        nunique_per_col = self.X_train.nunique()
        constant_columns = nunique_per_col[nunique_per_col == 1].index.tolist()

        self.logger.info(
            f"[GREEN]- Constant columns to be dropped: {constant_columns if len(constant_columns) > 0 else 'None'}"
        )

        # Drop constant columns from train, test, and val sets
        self.X_train = self.X_train.drop(columns=constant_columns)
        self.X_test = self.X_test.drop(
            columns=constant_columns, errors="ignore"
        )
        self.X_val = self.X_val.drop(columns=constant_columns, errors="ignore")
        if self.df_test is not None:
            self.df_test = self.df_test.drop(
                columns=constant_columns, errors="ignore"
            )

    def decide_outlier_imputation_order(
        self,
        column,
        missing_threshold=0.1,
        extreme_outlier_factor=2.0,
    ):
        """
        Decide whether to handle outliers before or after missing value imputation for a given column.

        Parameters:
            column (pd.Series): The column data (with missing values).
            get_threshold_method (callable): Function that returns 'iqr' or 'zscore' for the column.
            missing_threshold (float): Proportion of missing values to consider "high missingness".
            extreme_outlier_factor (float): Factor to amplify threshold bounds for detecting "extreme" outliers.

        Returns:
            str: 'before_imputation' or 'after_imputation'
        """
        data = column.dropna()
        total_count = len(column)
        missing_pct = (
            (total_count - len(data)) / total_count if total_count > 0 else 0
        )

        if len(data) < 3:
            # Not enough data to decide, default to after imputation
            return "after_imputation"

        threshold_method = self.get_threshold_method(data)

        # Calculate bounds for outlier detection, according to method
        if threshold_method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            # Standard bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Amplified bounds to detect extreme outliers
            extreme_lower = Q1 - extreme_outlier_factor * 1.5 * IQR
            extreme_upper = Q3 + extreme_outlier_factor * 1.5 * IQR

        elif threshold_method == "zscore":
            mean = data.mean()
            std = data.std()
            # Standard bounds
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            # Amplified bounds for extreme outliers
            extreme_lower = mean - extreme_outlier_factor * 3 * std
            extreme_upper = mean + extreme_outlier_factor * 3 * std

        else:
            # Unsupported method fallback
            return "after_imputation"

        # Detect outliers using the standard bounds:
        outliers = (data < lower_bound) | (data > upper_bound)
        outliers_count = outliers.sum()
        outliers_pct = outliers_count / len(data)

        # Detect extreme outliers using amplified bounds:
        extreme_outliers = (data < extreme_lower) | (data > extreme_upper)
        extreme_outliers_count = extreme_outliers.sum()

        # Decision logic:

        # Handle outliers BEFORE imputation if:
        # - The column has extreme outliers (significant number)
        # - OR the proportion of outliers is high and missingness is low (outliers probably errors)
        if extreme_outliers_count > 0:
            return "before_imputation"

        if outliers_pct > 0.1 and missing_pct < missing_threshold:
            # More than 10% outliers and missingness low means outliers likely distort
            return "before_imputation"

        # Otherwise, handle outliers after imputation
        return "after_imputation"

    def get_threshold_method(self, column, max_sample_size=5000):
        """
        Decide the outlier detection method for a single column (pandas Series) based on Shapiro-Wilk test,
        but only if sample size <= max_sample_size. Otherwise defaults to 'iqr'.

        Parameters:
            column (pd.Series): The data column to test.
            max_sample_size (int): Maximum allowed size to run Shapiro-Wilk. Defaults to 5000.

        Returns:
            str: 'zscore' if data approximately normal and sample size <= max_sample_size, 'iqr' otherwise.
        """
        # Drop NaN values before normality test
        data = column.dropna()

        # Use IQR if too few data points or too many
        if len(data) < 3 or len(data) > max_sample_size:
            return "iqr"  # Skip Shapiro for large or too small datasets

        stat, p_value = shapiro(data)

        if p_value > 0.05:
            return "zscore"
        else:
            return "iqr"

    def decide_outlier_handling_method(
        self,
        column,
        min_large_dataset=1000,
        max_drop_outlier_pct=0.05,
        missing_threshold=0.1,
        extreme_outlier_factor=2.0,
    ):
        """
        Decide how to handle outliers in a single column: 'drop', 'cap', or 'impute'.

        Parameters:
            column (pd.Series): Column data (with missing values).
            get_threshold_method (callable): Function returning 'iqr' or 'zscore' for the column.
            min_large_dataset (int): Threshold for dataset size to be considered large.
            max_drop_outlier_pct (float): Maximum % of outliers allowed to safely drop rows.
            missing_threshold (float): Threshold for missing value proportion considered high.
            extreme_outlier_factor (float): Multiplier to identify extreme outliers.

        Returns:
            str: One of 'drop', 'cap', or 'impute'.
        """
        data = column.dropna()
        n = len(data)
        missing_pct = (len(column) - n) / len(column) if len(column) > 0 else 0

        if n < 3:
            # Not enough data to decide meaningfully, default to 'impute'
            return "impute"

        threshold_method = self.get_threshold_method(data)

        # Calculate bounds for outliers
        if threshold_method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            extreme_lower = Q1 - extreme_outlier_factor * 1.5 * IQR
            extreme_upper = Q3 + extreme_outlier_factor * 1.5 * IQR

        elif threshold_method == "zscore":
            mean = data.mean()
            std = data.std()

            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            extreme_lower = mean - extreme_outlier_factor * 3 * std
            extreme_upper = mean + extreme_outlier_factor * 3 * std

        else:
            # Fallback, safest to cap
            return "cap"

        outliers = (data < lower_bound) | (data > upper_bound)
        outlier_pct = outliers.sum() / n if n > 0 else 0

        extreme_outliers = (data < extreme_lower) | (data > extreme_upper)
        extreme_outlier_count = extreme_outliers.sum()

        # Decision logic:

        # 1. If dataset is large (>= min_large_dataset),
        #    and outlier percentage is small enough (<= max_drop_outlier_pct),
        #    and there are some outliers → drop rows to clean data aggressively.
        if (
            (n >= min_large_dataset)
            and (outlier_pct <= max_drop_outlier_pct)
            and (outlier_pct > 0)
        ):
            return "drop"

        # 2. If there are extreme outliers, but dropping too many rows would lose data, cap them.
        if extreme_outlier_count > 0:
            return "cap"

        # 3. If missing data proportion is high or outlier % is moderate/large,
        #    imputing is safer to avoid losing data or bias.
        if (
            missing_pct > missing_threshold
            or outlier_pct > max_drop_outlier_pct
        ):
            return "impute"

        # 4. Default fallback to cap
        return "cap"

    def is_continuous_feature(self, column, unique_value_threshold=10):
        """
        Heuristic to recognize if a column is continuous numeric or categorical/count-like without domain knowledge.

        Parameters:
            column (pd.Series): The column to check.
            unique_value_threshold (int): Max unique values allowed to consider as categorical/count.

        Returns:
            bool: True if considered continuous numeric feature; False if likely categorical/count.
        """
        # If non-numeric dtype (object, category), treat as categorical
        infer_type = infer_dtype(column)
        if infer_type not in ["integer", "float"]:
            return False

        # If numeric but with very few unique values, treat as categorical/count
        # (e.g., integer counts, small discrete categories)
        unique_vals = column.dropna().unique()
        num_unique = len(unique_vals)
        if num_unique <= unique_value_threshold:
            return False

        # If mostly integers but many unique values, still treat continuous
        # e.g., float or int with > threshold unique values
        return True

    def handle_outliers(
        self, columns, before_or_after, method="", threshold_method=""
    ):
        """
        Automatically detect and handle outliers in self.X_train, and apply
        consistent changes to self.X_val and self.X_test.

        Args:
            method: str, one of ['drop', 'cap', 'impute']
            threshold_method: str, one of ['iqr', 'zscore']
        """
        self.logger.info(f"[GREEN]- Handling outliers {before_or_after}")
        X_train = self.X_train.copy()
        X_val = self.X_val.copy()
        X_test = self.X_test.copy()
        if self.df_test is not None:
            df_test = self.df_test.copy()
        else:
            df_test = None

        for col in X_train[columns].select_dtypes(include=[np.number]).columns:
            if self.is_continuous_feature(X_train[col]):
                if threshold_method == "":
                    cur_threshold_method = self.get_threshold_method(
                        X_train[col]
                    )
                else:
                    cur_threshold_method = threshold_method
                if cur_threshold_method == "iqr":
                    Q1 = X_train[col].quantile(0.25)
                    Q3 = X_train[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                elif cur_threshold_method == "zscore":
                    mean = X_train[col].mean()
                    std = X_train[col].std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                else:
                    raise ValueError("Unsupported threshold_method")

                # Identify outliers in train set
                outliers_train = (X_train[col] < lower_bound) | (
                    X_train[col] > upper_bound
                )

                if method == "":
                    cur_method = self.decide_outlier_handling_method(
                        X_train[col]
                    )
                else:
                    cur_method = method
                self.logger.debug(
                    f"{before_or_after}: {outliers_train.sum()} outliers found in {col}, threshold method: {cur_threshold_method}, outlier method: {cur_method}"
                )
                if outliers_train.sum() == 0:
                    continue
                if cur_method == "drop":
                    # Drop rows with outliers in train set
                    X_train = X_train.loc[~outliers_train]

                    # Also drop these rows in y_train if applicable
                    self.y_train = self.y_train.loc[X_train.index]

                    self.logger.debug(
                        f"[GREEN]- {outliers_train.sum()} outliers in column {col} dropped"
                    )
                elif cur_method == "cap":
                    X_train[col] = np.where(
                        X_train[col] < lower_bound, lower_bound, X_train[col]
                    )
                    X_train[col] = np.where(
                        X_train[col] > upper_bound, upper_bound, X_train[col]
                    )
                    X_val[col] = np.where(
                        X_val[col] < lower_bound, lower_bound, X_val[col]
                    )
                    X_val[col] = np.where(
                        X_val[col] > upper_bound, upper_bound, X_val[col]
                    )
                    X_test[col] = np.where(
                        X_test[col] < lower_bound, lower_bound, X_test[col]
                    )
                    X_test[col] = np.where(
                        X_test[col] > upper_bound, upper_bound, X_test[col]
                    )
                    if df_test is not None:
                        df_test[col] = np.where(
                            df_test[col] < lower_bound,
                            lower_bound,
                            df_test[col],
                        )
                        df_test[col] = np.where(
                            df_test[col] > upper_bound,
                            upper_bound,
                            df_test[col],
                        )
                    self.logger.debug(
                        f"[GREEN]  - {outliers_train.sum()} outliers in column {col} capped between {X_train[col].min()} and {X_train[col].max()}"
                    )
                elif cur_method == "impute":
                    median = X_train[col].median()
                    X_train.loc[outliers_train, col] = median
                    X_val.loc[
                        (X_val[col] < lower_bound) | (X_val[col] > upper_bound),
                        col,
                    ] = median
                    X_test.loc[
                        (X_test[col] < lower_bound)
                        | (X_test[col] > upper_bound),
                        col,
                    ] = median
                    if df_test is not None:
                        df_test.loc[
                            (df_test[col] < lower_bound)
                            | (df_test[col] > upper_bound),
                            col,
                        ] = median
                    self.logger.debug(
                        f"[GREEN]  - {outliers_train.sum()} outliers in column {col} imputed with median {median}"
                    )
                else:
                    raise ValueError("Unsupported handling method")

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        if df_test is not None:
            self.df_test = df_test

    def skip_outliers(self) -> bool:
        target_col = self.y_train
        unique_classes = target_col.unique()
        total_samples = len(target_col)
        if len(unique_classes) <= 5:
            for cls in unique_classes:
                freq = (target_col == cls).sum() / total_samples
                if freq < 0.01:
                    return True
        return False

    def get_feature_importances(self):
        X_train_enc = self.X_train.copy()

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
        y = self.y_train

        is_regression = infer_dtype(y) in {"integer", "float"}
        if np.issubdtype(y.dtype, np.number):
            # Numeric target: treat as regression
            is_regression = True
        else:
            # Non-numeric, treat as classification
            is_regression = False

        # Alternatively, you might check number of unique classes or task knowledge
        # Here a simple numeric dtype check is used

        # Fit appropriate model
        if is_regression:
            model = RandomForestRegressor(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)

        model.fit(X_train_enc, y)

        importances = dict(zip(X_train_enc.columns, model.feature_importances_))
        return importances

    def preprocess_missing_values(
        self,
        col_importance_thresh=0.01,
        col_missing_thresh=0.5,
        row_missing_thresh=0.05,
        max_row_drop_frac=0.1,
        skew_thresh=0.5,
    ):
        """
        Handles missing values by deciding to drop or impute based on X_train.
        Applies the same transformations to X_val and X_test.

        Parameters:
        - X_train, X_val, X_test: pd.DataFrame inputs
        - col_missing_thresh: fraction missing above which to drop the column
        - row_missing_thresh: fraction missing per row below which to drop rows
        - skew_thresh: absolute skewness above which a numerical feature is considered skewed
        - cat_cols: list of categorical column names (if None, inferred by dtype)

        Returns:
        - X_train_processed, X_val_processed, X_test_processed (pd.DataFrame)
        """

        X_train = self.X_train
        X_val = self.X_val
        X_test = self.X_test
        if self.df_test is not None:
            df_test = self.df_test
        else:
            df_test = None

        self.logger.info("[GREEN]- Processing missing values")
        feature_importances = self.get_feature_importances()
        # Step 1: Drop columns based on missingness and feature importance

        # Compute missingness per column
        missing_per_col = X_train.isna().mean()

        # Filter feature_importances to existing columns
        feature_importances = {
            k: v for k, v in feature_importances.items() if k in X_train.columns
        }

        # Normalize importance scores to sum to 1 (avoid zero sum)
        total_imp = sum(feature_importances.values())
        if total_imp == 0:
            norm_importances = {
                k: 1 / len(feature_importances) for k in feature_importances
            }
        else:
            norm_importances = {
                k: v / total_imp for k, v in feature_importances.items()
            }

        # Decide which columns to drop:
        cols_to_drop = []
        for col in X_train.columns:
            missingness = missing_per_col[col]
            importance = norm_importances.get(col, 0)
            # Drop column if missingness > threshold and importance < threshold
            if (
                missingness > col_missing_thresh
                and importance < col_importance_thresh
            ):
                cols_to_drop.append(col)
        if cols_to_drop != []:
            for col in cols_to_drop:
                self.logger.debug(
                    f"[GREEN]  - Dropping {col} because of too much missing values"
                )
        # Drop selected columns from all datasets
        X_train = X_train.drop(columns=cols_to_drop)
        X_val = X_val.drop(columns=cols_to_drop, errors="ignore")
        X_test = X_test.drop(columns=cols_to_drop, errors="ignore")
        if df_test is not None:
            df_test = df_test.drop(columns=cols_to_drop, errors="ignore")

        # Update feature importance dict accordingly
        feature_importances = {
            k: v
            for k, v in feature_importances.items()
            if k not in cols_to_drop
        }

        # Step 2: Drop rows based on weighted missingness and feature importance

        # Create missing mask for X_train (boolean DataFrame)
        missing_mask = X_train.isna()

        # Assign weights to missing values based on feature importance
        weights = pd.DataFrame(0, index=X_train.index, columns=X_train.columns)
        for col in X_train.columns:
            weights[col] = missing_mask[col].astype(
                float
            ) * norm_importances.get(col, 0)

        # Weighted missingness sum per row
        weighted_missing_per_row = weights.sum(axis=1)

        # Identify rows exceeding missingness threshold
        rows_to_drop = weighted_missing_per_row[
            weighted_missing_per_row > row_missing_thresh
        ].index

        fraction_rows_to_drop = len(rows_to_drop) / len(X_train)

        # Drop rows only if fraction is within allowable max
        if fraction_rows_to_drop <= max_row_drop_frac:
            X_train = X_train.drop(index=rows_to_drop)
            self.logger.debug(
                f"[GREEN]  - Dropped {len(rows_to_drop)} rows because of missing values."
            )
        else:
            # No rows dropped; keep all for imputation
            rows_to_drop = pd.Index([])

        # Step 3: Identify categorical columns
        cat_cols = []
        num_cols = []
        for col in X_train.columns:
            if infer_dtype(X_train[col]) in [
                "string",
                "category",
                "boolean",
                "object",
            ]:
                cat_cols.append(col)
            else:
                num_cols.append(col)

        # Step 4: Compute skewness on numerical cols
        skewness = X_train[num_cols].skew().abs()

        # Prepare imputers dict to hold imputers for each type
        imputers = {}

        # Numerical - split by skewness
        num_cols_low_skew = skewness[skewness <= skew_thresh].index.tolist()
        num_cols_high_skew = skewness[skewness > skew_thresh].index.tolist()

        # Imputer for numerical low skew with mean
        if num_cols_low_skew:
            imp_mean = SimpleImputer(strategy="mean")
            imp_mean.fit(X_train[num_cols_low_skew])
            imputers["num_mean"] = (imp_mean, num_cols_low_skew)
            for col in num_cols_low_skew:
                self.logger.debug(
                    f"[GREEN]  - Imputing missing values with 'mean'(low skewness numeric) for column {col}"
                )

        # Imputer for numerical high skew with median
        if num_cols_high_skew:
            imp_median = SimpleImputer(strategy="median")
            imp_median.fit(X_train[num_cols_high_skew])
            imputers["num_median"] = (imp_median, num_cols_high_skew)
            for col in num_cols_high_skew:
                self.logger.debug(
                    f"[GREEN]  - Imputing missing values with 'median'(high skewness numeric) for column {col}"
                )

        # Imputer for categorical with mode (most frequent)
        if cat_cols:
            imp_mode = SimpleImputer(strategy="most_frequent")
            imp_mode.fit(X_train[cat_cols])
            imputers["cat_mode"] = (imp_mode, cat_cols)
            for col in num_cols_high_skew:
                self.logger.debug(
                    f"[GREEN]  - Imputing missing values with 'mode'(category) for column {col}"
                )

        # Define a helper function to apply imputers consistently
        def apply_imputers(X):
            X = X.copy()
            for imp, cols in imputers.values():
                if cols:
                    X[cols] = imp.transform(X[cols])
            return X

        # Apply imputers to train/val/test
        self.X_train = apply_imputers(X_train)
        self.X_val = apply_imputers(X_val)
        self.X_test = apply_imputers(X_test)
        if self.df_test is not None:
            self.df_test = apply_imputers(df_test)

    def convert_column_to_boolean(self, column_name, is_target=False) -> bool:
        def is_boolean_series(series):
            # Check for boolean strings (case insensitive)
            str_vals = series.dropna().astype(str).str.lower().unique()
            bool_str_vals = {"true", "false"}
            # Check for numeric 0/1
            num_vals = set(series.dropna().unique())

            # Condition 1: strings are only "true" or "false"
            condition_str = all(val in bool_str_vals for val in str_vals)

            # Condition 2: Numeric values only 0 or 1
            condition_num = num_vals.issubset({0, 1})

            return condition_str or condition_num

        if not is_target:
            cur_series = self.X_train[column_name]
        else:
            cur_series = self.y_train
        if is_boolean_series(cur_series):
            # Convert the column in all datasets
            def to_bool(series):
                # If string values
                if series.dtype == object or series.dtype.name == "string":
                    return series.astype(str).str.lower() == "true"
                # Else numeric: convert 1 to True, 0 to False
                else:
                    return series.astype(bool)

            if not is_target:
                self.X_train[column_name] = to_bool(self.X_train[column_name])
                self.X_val[column_name] = to_bool(self.X_val[column_name])
                self.X_test[column_name] = to_bool(self.X_test[column_name])
                if self.df_test is not None:
                    self.df_test[column_name] = to_bool(
                        self.df_test[column_name]
                    )
            else:
                self.y_train = to_bool(self.y_train)
                self.y_val = to_bool(self.y_val)
                self.y_test = to_bool(self.y_test)
            self.logger.debug(
                f"[GREEN] - Column '{column_name}' converted to boolean."
            )
            return True
        else:
            return False

    def convert_column_to_category(self, column_name, is_target=False) -> bool:
        convert = False
        if not is_target:
            col = self.X_train[column_name]
        else:
            col = self.y_train
        max_unique = min(20, max(10, int(0.01 * len(col))))
        # Remove missing values for uniqueness check
        unique_vals = col.dropna().unique()

        if len(unique_vals) == 0:
            return False  # empty or all missing, skip

        # Case 1: Non-numeric and few unique values
        if not is_numeric_dtype(col) and len(unique_vals) <= max_unique:
            convert = True

        # Case2: Numeric but integer-like and few unique values
        elif is_numeric_dtype(col) and len(unique_vals) <= max_unique:
            # For float columns with integers + missing values, check integer-likeness
            if col.dtype == float:
                # Check if all non-NaN values are integer-like
                if col.dropna().apply(float.is_integer).all():
                    convert = True
            else:
                # integer dtype or others
                convert = True
        if convert:
            if not is_target:
                self.X_train[column_name] = col.astype("category")
                self.X_test[column_name] = self.X_test[column_name].astype(
                    "category"
                )
                self.X_val[column_name] = self.X_val[column_name].astype(
                    "category"
                )
                if self.df_test is not None:
                    self.df_test[column_name] = self.df_test[
                        column_name
                    ].astype("category")
            else:
                self.y_train = col.astype("category")
                self.y_test = self.y_test.astype("category")
                self.y_val = self.y_val.astype("category")

            self.logger.debug(
                f"[GREEN] - Column '{column_name}' converted to category."
            )
            return True
        else:
            return False

    def is_numeric_string_series(self, series):
        # Check if all non-null values can be converted to numbers
        # This allows int, float, scientific notation, negative sign, decimals
        # Return True if all non-null values are strings representing numbers

        # Drop missing values first
        s = series.dropna()

        # If empty after drop, return False (nothing to detect)
        if s.empty:
            return False

        # All values must be strings (or object dtype with strings)
        # Convert all to string (in case)
        s_str = s.astype(str)

        # Use pd.to_numeric with errors='coerce' to detect numeric strings
        numeric_converted = pd.to_numeric(s_str, errors="coerce")

        # If any non-null value failed conversion (NaN after to_numeric), then not all numeric strings
        return numeric_converted.notna().all()

    def convert_column_to_string(self, column_name, is_target=False) -> bool:
        if not is_target:
            col = self.X_train[column_name]
        else:
            col = self.y_train

        # If column is already numeric, no need to convert to string
        if is_numeric_dtype(col):
            return False

        # Check if column values are numeric strings
        if self.is_numeric_string_series(col):
            return False

        # Otherwise, convert to string dtype
        if not is_target:
            self.X_train[column_name] = col.astype("string")
            self.X_test[column_name] = self.X_test[column_name].astype("string")
            self.X_val[column_name] = self.X_val[column_name].astype("string")
            if self.df_test is not None:
                self.df_test[column_name] = self.df_test[column_name].astype(
                    "string"
                )
        else:
            self.y_train = col.astype("string")
            self.y_test = self.y_test.astype("string")
            self.y_val = self.y_val.astype("string")
        self.logger.debug(
            f"[GREEN] - Column '{column_name}' converted to string."
        )
        return True

    def convert_column_to_integer(self, column_name, is_target=False) -> bool:
        if not is_target:
            col = self.X_train[column_name]
        else:
            col = self.y_train
        if self.is_numeric_string_series(col):
            col = col.astype("float64")
        if col.apply(float.is_integer).all():
            if not is_target:
                self.X_train[column_name] = col.astype("int64")
                self.X_test[column_name] = self.X_test[column_name].astype(
                    "int64"
                )
                self.X_val[column_name] = self.X_val[column_name].astype(
                    "int64"
                )
                if self.df_test is not None:
                    self.df_test[column_name] = self.df_test[
                        column_name
                    ].astype("int64")
            else:
                self.y_train = col.astype("int64")
                self.y_test = self.y_test.astype("int64")
                self.y_val = self.y_val.astype("int64")
            return True
        return False

    def convert_column_to_float(self, column_name, is_target=False) -> bool:
        if not is_target:
            col = self.X_train[column_name]
        else:
            col = self.y_train[column_name]
        if self.is_numeric_string_series(col):
            if not is_target:
                self.X_train[column_name] = col.astype("float64")
                self.X_test[column_name] = self.X_test[column_name].astype(
                    "float64"
                )
                self.X_val[column_name] = self.X_val[column_name].astype(
                    "float64"
                )
                if self.df_test is not None:
                    self.df_test[column_name] = self.df_test[
                        column_name
                    ].astype("float64")
            else:
                self.y_train[column_name] = col.astype("float64")
                self.y_test[column_name] = self.y_test[column_name].astype(
                    "float64"
                )
                self.y_val = self.y_val.astype("float64")
            self.logger.debug(
                f"[GREEN] - Column '{column_name}' converted to float."
            )
            return True
        self.logger.error(f"Could not convert column {column_name}")
        return False

    def update_column_type(self, column_name: str, is_target=False) -> None:
        # check if boolean
        if self.convert_column_to_boolean(column_name, is_target):
            return
        if self.convert_column_to_category(column_name, is_target):
            return
        if self.convert_column_to_string(column_name, is_target):
            return
        if self.convert_column_to_integer(column_name, is_target):
            return
        self.convert_column_to_float(column_name, is_target)

    def preprocess(self):
        project = self.title if self.title else "dataset"
        self.logger.info(f"[MAGENTA]\nStarting preprocessing for {project}")
        result = self.eda.load_data()
        self.dict_description = self.eda.dict_description
        assert self.eda.df_train is not None
        if result != "":
            return result
        # self.eda.set_column_types(also_ints_and_bools=False)

        # 1 drop duplicate rows
        self.drop_duplicate_rows()
        self.target = self.eda.target

        # 2 split data 80/10/10 (train/test/validate)
        self.split_train_test_val()
        # eda object can be destroyed, everything is in this class now.
        del self.eda

        # 3 drop duplicate and constant columns
        self.drop_duplicate_columns()
        self.drop_constant_columns()
        skip_outliers = self.skip_outliers()
        if skip_outliers:
            self.logger.info(
                "[GREEN]- Leaving outliers as is due to target type (imbalanced classification)"
            )
        else:
            # 4 decide to handle outliers before or after dealing with missing values

            before_cols = []
            for col in self.X_train.select_dtypes(include=[np.number]).columns:
                if (
                    self.decide_outlier_imputation_order(
                        column=self.X_train[col]
                    )
                    == "before_imputation"
                ):
                    before_cols.append(col)

            # 4a handle outliers before dealing with missing values
            self.handle_outliers(
                columns=before_cols,
                before_or_after="before missing imputation",
                method="",
                threshold_method="",
            )
        # 5 missing values
        num_missing = self.X_train.isna().sum().sum()
        if num_missing > 0:
            self.preprocess_missing_values()
        else:

            self.logger.info("[GREEN]- No missing values to be processed.")
        # 5a handle outliers after dealing with missing values
        if not skip_outliers and num_missing > 0:
            self.handle_outliers(
                columns=self.X_train.select_dtypes(include=[np.number]).columns,
                before_or_after="after missing imputation",
                method="",
                threshold_method="",
            )

        # 6 update column types
        self.logger.info("[GREEN]- Updating column types.")
        for col in self.X_train.columns:
            self.update_column_type(col)
        self.update_column_type(self.target, is_target=True)

        # 7 encoding
        # 8 normalizing/scaling
        # 9 dim. reduction

        self.logger.info(f"[MAGENTA]Done preprocessing for {project}")
        self.post_process_eda()
        # self.X_train.to_csv(
        #     self.file_train.replace(".csv", "_prepro.csv"),
        #     index=False,
        # )
        return "Done! So far...."

    def post_process_eda(self):
        self.eda = AutoML_EDA(
            report_file=self.report_file,
            file_train=self.file_train,
            file_test=self.file_test,
            title=self.title,
            target=self.target,
            description=self.description,
            nogui=self.nogui,
            update_script=self.update_script,
            logger=self.logger,
        )
        self.eda.df_train = pd.concat([self.X_train, self.y_train], axis=1)
        self.eda.report_file = self.eda.report_file.replace(
            ".html", "-prepro.html"
        )
        self.eda.dict_description = self.dict_description
        self.eda.target = self.target
        self.eda.df_test = self.df_test
        self.eda.perform_eda(skip_load=True)
