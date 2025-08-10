# internal imports
from library import Logger, infer_dtype, check_classification
from modelling import AutoML_Modeling
from eda import AutoML_EDA

# external imports
from scipy.stats import shapiro
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    PowerTransformer,
    StandardScaler,
)
from typing import List, Tuple, Optional, Self, Literal
import warnings


class TargetTransformer:
    """
    Transforms the target variable using Yeo-Johnson power transform (to reduce skewness)
    followed by standard scaling (zero mean, unit variance). Supports inverse transformation.

    Attributes:
        pt (PowerTransformer): PowerTransformer instance using Yeo-Johnson method.
        scaler (StandardScaler): StandardScaler instance to scale the transformed target.
    """

    def __init__(self) -> None:
        """
        Initializes the TargetTransformer with PowerTransformer and StandardScaler.
        """
        self.pt: PowerTransformer = PowerTransformer(
            method="yeo-johnson", standardize=False
        )
        self.scaler: StandardScaler = StandardScaler()

    def fit(self, y_train: pd.Series | np.ndarray) -> Self:
        """
        Fits the power transformer and scaler on the training target data.

        Args:
            y_train (pd.Series | np.ndarray): Training target values.

        Returns:
            Self: The fitted transformer instance.
        """
        if isinstance(y_train, np.ndarray):
            y_train_reshaped = y_train.reshape(-1, 1)
        else:
            y_train_reshaped = np.asarray(y_train).reshape(-1, 1)

        y_transformed = self.pt.fit_transform(y_train_reshaped)
        self.scaler.fit(y_transformed)
        return self

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """
        Transforms the target data using the fitted power transformer and scaler.

        Args:
            y (pd.Series | np.ndarray): Target values to transform.

        Returns:
            np.ndarray: Transformed and scaled target data as 1D array.
        """
        if isinstance(y, np.ndarray):
            y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = np.asarray(y).reshape(-1, 1)
        y_transformed = self.pt.transform(y_reshaped)
        y_scaled = self.scaler.transform(y_transformed)
        return y_scaled.flatten()

    def inverse_transform(self, y_scaled: pd.Series | np.ndarray) -> np.ndarray:
        """
        Inverse transforms scaled target data back to the original scale.

        Args:
            y_scaled (pd.Series | np.ndarray): Scaled target values.

        Returns:
            np.ndarray: Original scale target values as 1D array.
        """
        if isinstance(y_scaled, pd.Series):
            y_scaled = np.asanyarray(y_scaled)
        y_transformed = self.scaler.inverse_transform(y_scaled.reshape(-1, 1))
        y_original = self.pt.inverse_transform(y_transformed)
        return y_original.flatten()


class AutoML_Preprocess:
    """
    Class that manages the preprocessing pipeline for an Automated Machine Learning application,
    including data loading, cleaning, splitting, missing value imputation, outlier handling,
    encoding, normalization, and target transformation.

    Attributes:
        report_file (str): Path for the preprocessing report.
        file_train (str): Path to the training data CSV file.
        file_test (str): Path to the optional test data CSV file.
        title (str): Title or project name.
        target (str): Name of target variable column.
        description (str): Description of the dataset/project.
        nogui (bool): Flag to indicate if GUI should be disabled.
        update_script (str): Path to a script to update.
        logger (Optional[Logger]): Logger instance.
        eda (AutoML_EDA): Exploratory Data Analysis instance.
        Others: Various attributes used in preprocessing pipeline.
    """

    def __init__(
        self,
        report_file: str,
        file_train: str,
        file_test: str = "",
        title: str = "",
        target: str = "",
        description: str = "",
        nogui: bool = True,
        update_script: str = "",
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Initializes the preprocessing pipeline with configuration and sets up logging and EDA.

        Args:
            report_file (str): Path to save the EDA report.
            file_train (str): Path to training dataset CSV.
            file_test (str, optional): Path to test dataset CSV. Defaults to "".
            title (str, optional): Project or dataset title. Defaults to "".
            target (str, optional): Target variable column name. Defaults to "".
            description (str, optional): Dataset/project description. Defaults to "".
            nogui (bool, optional): If True, disables GUI elements. Defaults to True.
            update_script (str, optional): Path to update script. Defaults to "".
            logger (Optional[Logger], optional): Logger instance for messages. If None,
                a default logger is instantiated. Defaults to None.
        """
        self.report_file: str = report_file
        self.file_train: str = file_train
        self.file_test: str = file_test
        self.title: str = title
        self.target: str = target
        self.description: str = description
        self.nogui: bool = nogui
        self.update_script: str = update_script
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger: Logger = logger
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

    def split_train_test_val(self) -> None:
        """
        Splits the original training data into training, validation, and test sets
        with proportions approximately 80% train, 10% validation, 10% test.

        The split is stratified based on target type:
        - If classification, stratify on target labels.
        - If regression, stratify by target quantile bins.

        Sets the following attributes:
            - self.X_train, self.X_val, self.X_test
            - self.y_train, self.y_val, self.y_test
            - self.df_test (optional test dataset)
        """
        assert self.eda.df_train is not None

        y: pd.Series = self.eda.df_train[self.target]
        X: pd.DataFrame = self.eda.df_train.drop(columns=[self.target])

        # Determine if target is classification or regression
        # Simple heuristic: if target is numeric with many unique values, treat as regression

        is_classification: bool = check_classification(target=y)

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

    def drop_duplicate_rows(self) -> None:
        """
        Removes duplicate rows from the original training dataframe `self.eda.df_train`.
        Logs the number of duplicates dropped and their indices.
        """
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

    def drop_duplicate_columns(self) -> None:
        """
        Identifies and removes duplicate columns in the training features `self.X_train`
        by comparing hash sums of each column. Drops the duplicate columns from all
        datasets (train, validation, test, and optional test set).

        Logs the list of dropped columns or 'None' if no duplicates were found.
        """
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

    def drop_constant_columns(self) -> None:
        """
        Detects columns in training features `self.X_train` with constant values
        (only one unique value) and drops them from all datasets (train, val, test,
        and optional test set).

        Logs which columns were dropped or 'None' if none were found.
        """
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
        column: pd.Series,
        missing_threshold: float = 0.1,
        extreme_outlier_factor: float = 2.0,
    ) -> Literal["after_imputation"] | Literal["before_imputation"]:
        """
        Decide whether outliers in a specific column should be handled before or after
        missing value imputation.

        The decision is based on:
            - Presence of extreme outliers.
            - Proportion of outliers relative to missing data.
            - Distribution shape estimated by IQR or Z-score methods.

        Args:
            column (pd.Series): Column data including missing values.
            missing_threshold (float): Missing value proportion threshold to consider low/high.
            extreme_outlier_factor (float): Factor to enlarge thresholds for extreme outliers.

        Returns:
            str: 'before_imputation' or 'after_imputation' indicating outlier handling timing.
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

    def get_threshold_method(
        self, column: pd.Series, max_sample_size: int = 5000
    ) -> Literal["iqr"] | Literal["zscore"]:
        """
        Determine the appropriate statistical method to detect outliers for a specific
        column. Uses the Shapiro-Wilk test for normality on sample sizes within
        the `max_sample_size` limit.

        Args:
            column (pd.Series): Data column for normality test.
            max_sample_size (int): Maximum size to run Shapiro-Wilk test.

        Returns:
            str: 'zscore' if data is approximately normal and sample size is manageable;
                'iqr' otherwise.
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
        column: pd.Series,
        min_large_dataset: int = 1000,
        max_drop_outlier_pct: float = 0.05,
        missing_threshold: float = 0.1,
        extreme_outlier_factor: float = 2.0,
    ) -> Literal["impute"] | Literal["cap"] | Literal["drop"]:
        """
        Decide how to treat outliers in a given column: drop rows, cap values, or impute.

        Logic depends on:
            - Dataset size.
            - Percent of outlier values.
            - Missing data proportion.
            - Presence of extreme outliers.

        Args:
            column (pd.Series): Column data with potential outliers.
            min_large_dataset (int): Dataset size for "large" classification.
            max_drop_outlier_pct (float): Max % outliers allowed for safe row dropping.
            missing_threshold (float): Threshold for high missingness to favor imputation.
            extreme_outlier_factor (float): Multiplier for extreme outlier detection.

        Returns:
            str: One of 'drop', 'cap', or 'impute' indicating chosen outlier handling method.
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

    def is_continuous_feature(
        self, column: pd.Series, unique_value_threshold: int = 10
    ) -> bool:
        """
        Heuristic to determine if a feature/column is continuous numeric or categorical/count.

        Args:
            column (pd.Series): Column to analyze.
            unique_value_threshold (int): Max unique values allowed to consider non-continuous.

        Returns:
            bool: True if column is continuous numeric feature, False if categorical or count-like.
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
        self,
        columns: List[str],
        before_or_after: str,
        method: str = "",
        threshold_method: str = "",
    ) -> None:
        """
        Detects and handles outliers on specified columns of the training dataset.
        Applies same transformations consistently on validation, test, and optional test datasets.

        Supported methods: 'drop' (drop rows), 'cap' (clip values), 'impute' (median imputation).
        Supports threshold methods: 'iqr' or 'zscore'.

        Args:
            columns (List[str]): Columns to handle outliers on.
            before_or_after (str): Description of whether handling is before or after imputation.
            method (str): Optional explicit method for handling outliers; if empty auto-decides.
            threshold_method (str): Optional choice of threshold method; if empty auto-decides.
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
        """
        Decide whether to skip outlier handling based on target distribution.

        This heuristic checks for imbalanced classification problems by
        examining the number of unique classes in the target and the frequency
        of the rarest classes.

        Returns:
            bool: True if outlier handling should be skipped due to imbalanced classes,
                False otherwise.
        """
        target_col = self.y_train
        unique_classes = target_col.unique()
        total_samples = len(target_col)
        if len(unique_classes) <= 5:
            for cls in unique_classes:
                freq = (target_col == cls).sum() / total_samples
                if freq < 0.01:
                    return True
        return False

    def get_feature_importances(self) -> dict[str, np.ndarray]:
        """
        Compute feature importances using a Random Forest model fit on encoded training data.

        Encodes categorical features using OrdinalEncoder, fits a classifier or regressor
        based on the target type, and returns feature importance scores.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping feature names to importance scores.
        """
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

        is_regression = not check_classification(y)

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
        col_importance_thresh: float = 0.01,
        col_missing_thresh: float = 0.5,
        row_missing_thresh: float = 0.05,
        max_row_drop_frac: float = 0.1,
        skew_thresh: float = 0.5,
    ) -> None:
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

    def convert_column_to_boolean(self, column_name: str) -> bool:
        """
        Attempt to convert a column to a boolean 0/1 integer type.

        Recognizes columns that contain only boolean literals ('true', 'false')
        or numeric 0/1 values and converts them to integer 0/1 for all data splits.

        Args:
            column_name (str): The name of the column to convert.

        Returns:
            bool: True if conversion was successful, False otherwise.
        """

        def is_boolean_series(series: pd.Series) -> bool:
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

        cur_series = self.X_train[column_name]

        if is_boolean_series(cur_series):
            # Convert the column in all datasets
            def to_bool(series: pd.Series) -> pd.Series:
                # If string values
                if series.dtype == object or series.dtype.name == "string":
                    return series.astype(str).str.lower() == "true"
                # Else numeric: convert 1 to True, 0 to False
                else:
                    return series.astype(bool)

            self.X_train[column_name] = to_bool(
                self.X_train[column_name]
            ).astype(int)
            self.X_val[column_name] = to_bool(self.X_val[column_name]).astype(
                int
            )
            self.X_test[column_name] = to_bool(self.X_test[column_name]).astype(
                int
            )
            if self.df_test is not None:
                self.df_test[column_name] = to_bool(
                    self.df_test[column_name]
                ).astype(int)

            self.logger.debug(
                f"[GREEN] - Column '{column_name}' converted to (0/1) boolean."
            )
            return True
        else:
            return False

    def convert_column_to_category(self, column_name: str) -> bool:
        """
        Convert a column to pandas category type if it qualifies as categorical.

        Criteria:
        - Non-numeric columns with fewer unique values than a dynamic threshold.
        - Numeric columns which contain integer-like values with low cardinality.

        Converts the column in all splits (train, val, test, optional test).

        Args:
            column_name (str): The name of the column to convert.

        Returns:
            bool: True if converted, False otherwise.
        """
        convert = False
        col = self.X_train[column_name]

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
            self.X_train[column_name] = col.astype("category")
            self.X_test[column_name] = self.X_test[column_name].astype(
                "category"
            )
            self.X_val[column_name] = self.X_val[column_name].astype("category")
            if self.df_test is not None:
                self.df_test[column_name] = self.df_test[column_name].astype(
                    "category"
                )
            self.logger.debug(
                f"[GREEN] - Column '{column_name}' converted to category."
            )
            return True
        else:
            return False

    def is_numeric_string_series(self, series: pd.Series) -> bool:
        """
        Check if a pandas Series of strings represents numeric values for all non-null entries.

        Uses pandas `to_numeric` with error coercion to detect if all entries can be converted.

        Args:
            series (pd.Series): Series to check.

        Returns:
            bool: True if all non-null values are numeric strings, False otherwise.
        """
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
        return bool(numeric_converted.notna().all())

    def convert_column_to_string(self, column_name: str) -> bool:
        """
        Convert a column to pandas 'string' dtype if it is not numeric or numeric strings.

        Does nothing if column is numeric or numeric strings to avoid incorrect conversions.

        Args:
            column_name (str): The name of the column to convert.

        Returns:
            bool: True if converted to string, False otherwise.
        """
        col = self.X_train[column_name]

        # If column is already numeric, no need to convert to string
        if is_numeric_dtype(col):
            return False

        # Check if column values are numeric strings
        if self.is_numeric_string_series(col):
            return False

        # Otherwise, convert to string dtype

        self.X_train[column_name] = col.astype("string")
        self.X_test[column_name] = self.X_test[column_name].astype("string")
        self.X_val[column_name] = self.X_val[column_name].astype("string")
        if self.df_test is not None:
            self.df_test[column_name] = self.df_test[column_name].astype(
                "string"
            )
        self.logger.debug(
            f"[GREEN] - Column '{column_name}' converted to string."
        )
        return True

    def convert_column_to_integer(self, column_name: str) -> bool:
        """
        Convert a column to integer dtype if all values are integer-like.

        Converts numeric strings to float first if needed, then to int64 if all values are whole numbers.

        Args:
            column_name (str): The name of the column to convert.

        Returns:
            bool: True if conversion is done, False otherwise.
        """
        col = self.X_train[column_name]

        if self.is_numeric_string_series(col):
            col = col.astype("float64")
        if col.apply(float.is_integer).all():

            self.X_train[column_name] = col.astype("int64")
            self.X_test[column_name] = self.X_test[column_name].astype("int64")
            self.X_val[column_name] = self.X_val[column_name].astype("int64")
            if self.df_test is not None:
                self.df_test[column_name] = self.df_test[column_name].astype(
                    "int64"
                )
        return False

    def convert_column_to_float(self, column_name: str) -> bool:
        """
        Convert a column to float64 dtype if the column contains numeric strings.

        Args:
            column_name (str): The name of the column to convert.

        Returns:
            bool: True if conversion is successful, False otherwise.
        """
        col = self.X_train[column_name]

        if self.is_numeric_string_series(col):

            self.X_train[column_name] = col.astype("float64")
            self.X_test[column_name] = self.X_test[column_name].astype(
                "float64"
            )
            self.X_val[column_name] = self.X_val[column_name].astype("float64")
            if self.df_test is not None:
                self.df_test[column_name] = self.df_test[column_name].astype(
                    "float64"
                )
            self.logger.debug(
                f"[GREEN] - Column '{column_name}' converted to float."
            )
            return True
        self.logger.error(f"Could not convert column {column_name}")
        return False

    def update_column_type(self, column_name: str) -> None:
        """
        Update the data type of a single column by attempting conversions in sequence.

        Conversion attempts in order:
            - Boolean (0/1 integer)
            - Category (pandas category)
            - String (pandas string)
            - Float64 numeric

        Args:
            column_name (str): The column to update.
        """
        # check if boolean
        if self.convert_column_to_boolean(column_name):
            return
        if self.convert_column_to_category(column_name):
            return
        if self.convert_column_to_string(column_name):
            return
        # if self.convert_column_to_integer(column_name):
        #     return
        self.convert_column_to_float(column_name)

    def auto_encode_features(
        self, max_unique_for_categorical: int = 15
    ) -> dict[str, OneHotEncoder | OrdinalEncoder]:
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
            encoders = {}
            transformed_train = self.X_train.copy()
            transformed_val = self.X_val.copy()
            transformed_test = self.X_test.copy()
            transformed_df_test = None
            if self.df_test is not None:
                transformed_df_test = self.df_test.copy()
            df = self.X_train.copy()

            for col in df.columns:
                self.logger.debug(f"{col}")
                col_data = df[col]
                unique_vals = col_data.nunique()
                dtype = col_data.dtype

                # Boolean: convert to int (0/1)
                if pd.api.types.is_bool_dtype(dtype):
                    transformed_train[col] = col_data.astype(int)
                    transformed_val[col] = self.X_val[col].astype(int)
                    transformed_test[col] = self.X_test[col].astype(int)
                    if (
                        self.df_test is not None
                        and transformed_df_test is not None
                    ):
                        transformed_df_test[col] = self.df_test[col].astype(int)
                    continue

                # Category dtype: nominal categorical -> OneHotEncoder with handle_unknown='ignore'
                elif isinstance(dtype, pd.CategoricalDtype):
                    self.logger.debug(f"{col}: OneHotEncoder")
                    encoder = OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    )
                    encoded_train = encoder.fit_transform(
                        col_data.values.reshape(-1, 1)
                    )
                    cats = encoder.categories_[0]
                    cols_one_hot = [f"{col}_{cat}" for cat in cats[1:]]  # type: ignore
                    encoded_train_df = pd.DataFrame(
                        encoded_train, columns=cols_one_hot, index=df.index
                    )
                    transformed_train = transformed_train.drop(
                        columns=[col]
                    ).join(encoded_train_df)

                    encoded_val = encoder.transform(
                        self.X_val[col].values.reshape(-1, 1)
                    )
                    encoded_val_df = pd.DataFrame(
                        encoded_val, columns=cols_one_hot, index=self.X_val.index  # type: ignore
                    )
                    transformed_val = transformed_val.drop(columns=[col]).join(
                        encoded_val_df
                    )

                    encoded_test = encoder.transform(
                        self.X_test[col].values.reshape(-1, 1)
                    )
                    encoded_test_df = pd.DataFrame(
                        encoded_test, columns=cols_one_hot, index=self.X_test.index  # type: ignore
                    )
                    transformed_test = transformed_test.drop(
                        columns=[col]
                    ).join(encoded_test_df)
                    if (
                        self.df_test is not None
                        and transformed_df_test is not None
                    ):
                        encoded_df_test = encoder.transform(
                            self.df_test[col].values.reshape(-1, 1)
                        )
                        encoded_df_test_df = pd.DataFrame(
                            encoded_df_test, columns=cols_one_hot, index=self.df_test.index  # type: ignore
                        )
                        transformed_df_test = transformed_df_test.drop(
                            columns=[col]
                        ).join(encoded_df_test_df)
                    encoders[col] = encoder
                    continue

                # String/object dtype: treat as nominal categorical with OneHotEncoder and handle_unknown='ignore'
                elif pd.api.types.is_object_dtype(dtype):
                    self.logger.debug(f"{col}: OneHotEncoder")
                    temp_cat = col_data.astype("category")
                    encoder = OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    )
                    encoded_train = encoder.fit_transform(
                        temp_cat.values.reshape(-1, 1)
                    )
                    cats = encoder.categories_[0]
                    cols_one_hot = [f"{col}_{cat}" for cat in cats[1:]]  # type: ignore
                    encoded_train_df = pd.DataFrame(
                        encoded_train, columns=cols_one_hot, index=df.index
                    )
                    transformed_train = transformed_train.drop(
                        columns=[col]
                    ).join(encoded_train_df)

                    encoded_val = encoder.transform(
                        self.X_val[col].astype("category").values.reshape(-1, 1)
                    )
                    encoded_val_df = pd.DataFrame(
                        encoded_val, columns=cols_one_hot, index=self.X_val.index  # type: ignore
                    )
                    transformed_val = transformed_val.drop(columns=[col]).join(
                        encoded_val_df
                    )

                    encoded_test = encoder.transform(
                        self.X_test[col]
                        .astype("category")
                        .values.reshape(-1, 1)
                    )
                    encoded_test_df = pd.DataFrame(
                        encoded_test, columns=cols_one_hot, index=self.X_test.index  # type: ignore
                    )
                    transformed_test = transformed_test.drop(
                        columns=[col]
                    ).join(encoded_test_df)
                    if (
                        self.df_test is not None
                        and transformed_df_test is not None
                    ):
                        encoded_df_test = encoder.transform(
                            self.df_test[col]
                            .astype("category")
                            .values.reshape(-1, 1)
                        )
                        encoded_df_test_df = pd.DataFrame(
                            encoded_df_test, columns=cols_one_hot, index=self.df_test.index  # type: ignore
                        )
                        transformed_df_test = transformed_df_test.drop(
                            columns=[col]
                        ).join(encoded_df_test_df)
                    encoders[col] = encoder
                    continue

                # Numeric dtype with low cardinality: one-hot or ordinal encoding with unknown category handling
                elif pd.api.types.is_numeric_dtype(dtype):
                    if unique_vals <= max_unique_for_categorical:
                        if self.y_train is not None:
                            df_combined = df.copy()
                            df_combined[self.y_train.name] = self.y_train
                            means = df_combined.groupby(col)[
                                self.y_train.name
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
                                        col_data.values.reshape(-1, 1)
                                    )
                                )
                                transformed_val[[col]] = encoder.transform(
                                    self.X_val[col].values.reshape(-1, 1)
                                )
                                transformed_test[[col]] = encoder.transform(
                                    self.X_test[col].values.reshape(-1, 1)
                                )
                                if (
                                    self.df_test is not None
                                    and transformed_df_test is not None
                                ):
                                    transformed_df_test[[col]] = (
                                        encoder.transform(
                                            self.df_test[col].values.reshape(
                                                -1, 1
                                            )
                                        )
                                    )
                                self.logger.debug(
                                    f"{col}: OrdinalEncoder with unknown category handling"
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
                                    col_data.values.reshape(-1, 1)
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

                                encoded_val = encoder.transform(
                                    self.X_val[col].values.reshape(-1, 1)
                                )
                                encoded_val_df = pd.DataFrame(
                                    encoded_val,  # type: ignore
                                    columns=cols_one_hot,
                                    index=self.X_val.index,
                                )
                                transformed_val = transformed_val.drop(
                                    columns=[col]
                                ).join(encoded_val_df)

                                encoded_test = encoder.transform(
                                    self.X_test[col].values.reshape(-1, 1)
                                )
                                encoded_test_df = pd.DataFrame(
                                    encoded_test,  # type: ignore
                                    columns=cols_one_hot,
                                    index=self.X_test.index,
                                )
                                transformed_test = transformed_test.drop(
                                    columns=[col]
                                ).join(encoded_test_df)
                                if (
                                    self.df_test is not None
                                    and transformed_df_test is not None
                                ):
                                    encoded_df_test = encoder.transform(
                                        self.df_test[col].values.reshape(-1, 1)
                                    )
                                    encoded_df_test_df = pd.DataFrame(
                                        encoded_df_test,  # type: ignore
                                        columns=cols_one_hot,
                                        index=self.df_test.index,
                                    )
                                    transformed_df_test = (
                                        transformed_df_test.drop(
                                            columns=[col]
                                        ).join(encoded_df_test_df)
                                    )
                                self.logger.debug(
                                    f"{col}: OneHotEncoder with unknown category handling"
                                )
                                encoders[col] = encoder
                        else:
                            encoder = OneHotEncoder(
                                drop="first",
                                sparse_output=False,
                                handle_unknown="ignore",
                            )
                            encoded_train = encoder.fit_transform(
                                col_data.values.reshape(-1, 1)
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

                            encoded_val = encoder.transform(
                                self.X_val[col].values.reshape(-1, 1)
                            )
                            encoded_val_df = pd.DataFrame(
                                encoded_val,  # type: ignore
                                columns=cols_one_hot,
                                index=self.X_val.index,
                            )
                            transformed_val = transformed_val.drop(
                                columns=[col]
                            ).join(encoded_val_df)

                            encoded_test = encoder.transform(
                                self.X_test[col].values.reshape(-1, 1)
                            )
                            encoded_test_df = pd.DataFrame(
                                encoded_test,  # type: ignore
                                columns=cols_one_hot,
                                index=self.X_test.index,
                            )
                            transformed_test = transformed_test.drop(
                                columns=[col]
                            ).join(encoded_test_df)
                            if (
                                self.df_test is not None
                                and transformed_df_test is not None
                            ):
                                encoded_df_test = encoder.transform(
                                    self.df_test[col].values.reshape(-1, 1)
                                )
                                encoded_df_test_df = pd.DataFrame(
                                    encoded_df_test,  # type: ignore
                                    columns=cols_one_hot,
                                    index=self.df_test.index,
                                )
                                transformed_df_test = transformed_df_test.drop(
                                    columns=[col]
                                ).join(encoded_df_test_df)
                            self.logger.debug(
                                f"{col}: OneHotEncoder with unknown category handling"
                            )
                            encoders[col] = encoder
                    else:
                        # Leave continuous numeric unchanged
                        pass

        self.X_train = transformed_train
        self.X_val = transformed_val
        self.X_test = transformed_test
        if self.df_test is not None and transformed_df_test is not None:
            self.df_test = transformed_df_test
        return encoders

    def encode_targets(self) -> None:
        """
        Encode the target variables (y_train, y_val, y_test) if they are non-numeric.

        Uses LabelEncoder to convert string targets to numeric labels.
        Converts boolean targets to integers.

        No operation if targets are already numeric.
        """
        le = LabelEncoder()
        if not pd.api.types.is_numeric_dtype(self.y_train):
            # Fit on y_train
            le.fit(self.y_train)
            # Transform y_train, y_test, y_val using the same encoder
            self.y_train = pd.Series(
                le.transform(self.y_train),  # type: ignore
                index=self.y_train.index,
                name=self.y_train.name,
            )
            self.y_test = pd.Series(
                le.transform(self.y_test),  # type: ignore
                index=self.y_test.index,
                name=self.y_test.name,
            )
            self.y_val = pd.Series(
                le.transform(self.y_val),  # type: ignore
                index=self.y_val.index,
                name=self.y_val.name,
            )
        elif pd.api.types.is_bool_dtype(self.y_train):
            self.y_train = self.y_train.astype(int)
            self.y_test = self.y_test.astype(int)
            self.y_val = self.y_val.astype(int)

    def drop_strings(self) -> None:
        """
        Drop any feature columns in training, validation, test, and optional test datasets
        that are detected as string dtype.

        This is often done to remove unprocessable text columns before modeling.
        """
        # Identify columns with string dtype (including object dtype with strings)
        drop_cols = [
            col
            for col in self.X_train.columns
            if pd.api.types.is_string_dtype(self.X_train[col])
        ]
        self.logger.debug(f"[GREEN]- dropping {drop_cols} as they are strings")
        if drop_cols:
            self.X_train.drop(columns=drop_cols, inplace=True)
            self.X_val.drop(columns=drop_cols, inplace=True)
            self.X_test.drop(columns=drop_cols, inplace=True)
            if self.df_test is not None:
                self.df_test.drop(columns=drop_cols, inplace=True)

    def standardize_column(
        self,
        column_name: str,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        df_test: Optional[pd.DataFrame],
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]
    ]:
        """
        Standardize a single column to zero mean and unit variance across train, val, test, and optional test sets.

        Fits StandardScaler on the training column, then applies the same transformation to all sets.

        Args:
            column_name (str): The column to standardize.
            X_train (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            X_test (pd.DataFrame): Test features.
            df_test (Optional[pd.DataFrame]): Optional external test set.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: Transformed datasets.
        """
        scaler = StandardScaler()

        # Fit scaler on train column (reshape to 2D)
        train_col = X_train[[column_name]]
        scaler.fit(train_col)

        # Transform column in each dataset
        X_train.loc[:, column_name] = scaler.transform(train_col)
        X_val.loc[:, column_name] = scaler.transform(X_val[[column_name]])
        X_test.loc[:, column_name] = scaler.transform(X_test[[column_name]])
        if df_test is not None:
            df_test.loc[:, column_name] = scaler.transform(
                df_test[[column_name]]
            )

        return X_train, X_val, X_test, df_test

    def normalize_columns(self, skewness_threshold: float = 0.75) -> None:
        """
        Apply power transform (Yeo-Johnson) to skewed numeric columns and standard scale all numeric columns.

        Columns with absolute skewness higher than `skewness_threshold` are transformed.
        Scaling is fit on training set and applied to validation, test, and optional test.

        Args:
            skewness_threshold (float): Skewness level beyond which to apply Yeo-Johnson transform.
        """
        self.logger.info("[GREEN]- Normalizing")
        for column_name in self.X_train.columns:
            if column_name in self.encoders:
                # skip encoded columns
                continue
            skewness_value = self.X_train[column_name].skew()
            if abs(skewness_value) > skewness_threshold:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                # Fit on train column reshaped as 2D array
                X_train_col = self.X_train[
                    [column_name]
                ]  # keeps DataFrame format for transform
                pt.fit(X_train_col)

                # Transform all three sets' column
                self.X_train[column_name] = pt.transform(X_train_col)
                self.X_val[column_name] = pt.transform(
                    self.X_val[[column_name]]
                )
                self.X_test[column_name] = pt.transform(
                    self.X_test[[column_name]]
                )
                if self.df_test is not None:
                    self.df_test[column_name] = pt.transform(
                        self.df_test[[column_name]]
                    )
                self.logger.debug(
                    f"- Skewness found for {column_name}, yeo-johnson transormer applied"
                )
            scaler = StandardScaler()
            train_col = self.X_train[[column_name]]
            scaler.fit(train_col)

            # Transform returns 2D array, extract to 1D to assign safely
            self.X_train.loc[:, column_name] = (
                scaler.transform(train_col).flatten().astype(np.float64)
            )
            self.X_val.loc[:, column_name] = (
                scaler.transform(self.X_val[[column_name]])
                .flatten()
                .astype(np.float64)
            )
            self.X_test.loc[:, column_name] = (
                scaler.transform(self.X_test[[column_name]])
                .flatten()
                .astype(np.float64)
            )
            if self.df_test is not None:
                self.df_test.loc[:, column_name] = (
                    scaler.transform(self.df_test[[column_name]])
                    .flatten()
                    .astype(np.float64)
                )

    def standardize_target(self) -> None:
        """
        Fit a TargetTransformer using Yeo-Johnson transform and standard scaling on y_train,
        then transform y_train, y_val, and y_test targets correspondingly.

        Stores the target_transformer object for later inverse transformations if needed.
        """
        # 1. Fit transformer on training target
        target_transformer = TargetTransformer()
        target_transformer.fit(self.y_train)

        # 2. Transform target in train, val, test sets
        self.y_train = pd.Series(
            target_transformer.transform(self.y_train),
            index=self.y_train.index,
            name=self.y_train.name,
        )
        self.y_val = pd.Series(
            target_transformer.transform(self.y_val),
            index=self.y_val.index,
            name=self.y_val.name,
        )
        self.y_test = pd.Series(
            target_transformer.transform(self.y_test),
            index=self.y_test.index,
            name=self.y_test.name,
        )
        self.target_transformer = target_transformer

    def preprocess(self) -> str:
        """
        Run the entire preprocessing pipeline:

        1. Load data from files.
        2. Drop duplicate rows.
        3. Split data into training, validation and test sets (80/10/10).
        4. Drop duplicate and constant columns.
        5. Optionally handle outliers before missing value imputation.
        6. Handle missing values by dropping and imputing.
        7. Optionally handle outliers after missing value imputation.
        8. Update column data types to best guesses.
        9. Encode target variables.
        10. Drop string columns.
        11. Auto-encode features.
        12. Normalize and scale features.
        13. Normalize target variable if regression.
        14. Save preprocessed train dataset to CSV.
        15. Post-process EDA reports for preprocessed dataset.

        Returns:
            str: Status message after preprocessing finishes or error message.
        """
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
        skip_outliers: bool = self.skip_outliers()
        if skip_outliers:
            self.logger.info(
                msg="[GREEN]- Leaving outliers as is due to target type (imbalanced classification)"
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
        self.encode_targets()
        self.drop_strings()
        # 7 encoding
        self.encoders = self.auto_encode_features()
        # 8 normalizing/scaling
        self.normalize_columns()
        # 9 normalize target
        if not check_classification(self.y_train):
            self.standardize_target()
        # 10 dim. reduction

        self.logger.info(f"[MAGENTA]Done preprocessing for {project}")
        # self.post_process_eda()
        self.model = AutoML_Modeling(
            X_train=self.X_train,
            X_val=self.X_val,
            X_test=self.X_test,
            y_train=self.y_train,
            y_val=self.y_val,
            y_test=self.y_test,
            df_test=self.df_test,
            logger=self.logger,
        )

        self.X_train.to_csv(
            self.file_train.replace(".csv", "_prepro.csv"),
            index=False,
        )
        return "Done! So far...."

    def post_process_eda(self) -> None:
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
        """
        Regenerate the exploratory data analysis (EDA) object for the preprocessed train data.

        This updates the EDA report file path to indicate preprocessed data,
        copies over dataset description and target information.

        Does not reload data from file but works on preprocessed datasets in memory.
        """
        self.eda.df_train = pd.concat([self.X_train, self.y_train], axis=1)
        self.eda.report_file = self.eda.report_file.replace(
            ".html", "-prepro.html"
        )
        self.eda.dict_description = self.dict_description
        self.eda.target = self.target
        self.eda.df_test = self.df_test
        self.eda.perform_eda(skip_load=True)
