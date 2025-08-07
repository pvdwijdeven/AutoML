from library import (
    Logger,
)
from eda import AutoML_EDA
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
import numpy as np
from library import infer_dtype
import pandas as pd


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
        is_classification = False
        if pd.api.types.is_numeric_dtype(y):
            unique_vals = y.nunique()
            if (
                unique_vals < 20
            ):  # threshold for unique classes (tune as needed)
                is_classification = True
        else:
            # Non-numeric targets are treated as classification
            is_classification = True

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

        X_train = self.X_train.copy()
        X_val = self.X_val.copy()
        X_test = self.X_test.copy()

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

                    # For val/test sets, drop rows that would be outliers by same bounds
                    outliers_val = (X_val[col] < lower_bound) | (
                        X_val[col] > upper_bound
                    )
                    X_val = X_val.loc[~outliers_val]
                    self.y_val = self.y_val.loc[X_val.index]

                    outliers_test = (X_test[col] < lower_bound) | (
                        X_test[col] > upper_bound
                    )
                    X_test = X_test.loc[~outliers_test]
                    self.y_test = self.y_test.loc[X_test.index]
                    self.logger.info(
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
                    self.logger.info(
                        f"[GREEN]- {outliers_train.sum()} outliers in column {col} capped between {X_train[col].min()} and {X_train[col].max()}"
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
                    self.logger.info(
                        f"[GREEN]- {outliers_train.sum()} outliers in column {col} imputed with median {median}"
                    )
                else:
                    raise ValueError("Unsupported handling method")

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

    def skip_outliers(self) -> bool:
        target_col = self.y_train
        # Check if target's dtype is categorical
        if infer_dtype(target_col) in ["category", "boolean"]:
            unique_classes = target_col.unique()
            total_samples = len(target_col)
            if len(unique_classes) <= 5:
                for cls in unique_classes:
                    freq = (target_col == cls).sum() / total_samples
                    if freq < 0.01:
                        return True
        return False

    def preprocess(self):
        project = self.title if self.title else "dataset"
        self.logger.info(f"[MAGENTA]\nStarting preprocessing for {project}")
        result = self.eda.load_data()
        assert self.eda.df_train is not None
        if result != "":
            return result
        self.eda.set_column_types(also_ints=False)

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
            self.logger.warning(
                "Leaving outliers as is due to target type (imbalanced classification)"
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
            pass
        # 5a handle outliers after dealing with missing values
        if not skip_outliers and num_missing > 0:
            self.handle_outliers(
                columns=self.X_train.select_dtypes(include=[np.number]).columns,
                before_or_after="after missing imputation",
                method="",
                threshold_method="",
            )
        # 6 encoding
        # 7 normalizing/scaling
        # 8 dim. reduction
        self.logger.info("[MAGENTA] I am done here...")
        return "Done! So far...."
