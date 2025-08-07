from library import (
    Logger,
)

from eda import AutoML_EDA
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro


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
            self.report_file,
            self.file_train,
            self.file_test,
            self.title,
            self.target,
            self.description,
            self.nogui,
            self.update_script,
            self.logger,
        )

    def split_train_test_val(self):
        assert self.eda.df_train is not None
        # First split: train and temp (validation + test)
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            self.eda.df_train.drop(columns=[self.target]),
            self.eda.df_train[self.target],
            test_size=0.2,
            random_state=42,
            stratify=self.eda.df_train[self.target],
        )
        # Second split: split temp into validation and test (each 10% if test_size=0.5 here)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        self.df_test = self.eda.df_test
        self.logger.info(
            f"[GREEN]- Split train/test/validation ({self.X_train.shape[0]}/{self.X_test.shape[0]}/{self.X_val.shape[0]} rows)"
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
        self.logger.info("[GREEN]- Dropped no duplicate rows.")
        if num_rows_after < num_rows_before:
            self.logger.info(
                f"[GREEN]- Dropped {num_rows_before - num_rows_after} duplicate rows."
            )
            self.logger.info(
                f"[GREEN]  Dropped rows with indices: {dropped_indices}"
            )

    def drop_duplicate_columns(self):
        duplicate_columns = self.X_train.columns[self.X_train.T.duplicated()]
        self.logger.info(
            f"[GREEN] - Duplicate columns to be dropped: {list(duplicate_columns) if len(list(duplicate_columns))>0 else 'None'}"
        )
        # Drop duplicate columns in self.df_train
        self.X_train = self.X_train.loc[:, ~self.X_train.T.duplicated()]
        # Drop the same duplicate columns from self.X_test and self.X_val
        self.X_test = self.X_test.drop(
            columns=duplicate_columns, errors="ignore"
        )
        self.X_val = self.X_val.drop(columns=duplicate_columns, errors="ignore")

    def drop_constant_columns(self):
        # Identify columns in X_train that have the same value in every row
        constant_columns = [
            col
            for col in self.X_train.columns
            if self.X_train[col].nunique() == 1
        ]

        self.logger.info(
            f"[GREEN] - Constant columns to be dropped: {constant_columns if len(constant_columns)>0 else 'None'}"
        )

        # Drop these constant columns from X_train
        self.X_train = self.X_train.drop(columns=constant_columns)

        # Drop the same columns from X_test and X_val (ignore errors if some col don't exist)
        self.X_test = self.X_test.drop(
            columns=constant_columns, errors="ignore"
        )
        self.X_val = self.X_val.drop(columns=constant_columns, errors="ignore")

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

    def handle_outliers(self, method="cap", threshold_method=""):
        """
        Automatically detect and handle outliers in self.X_train, and apply
        consistent changes to self.X_val and self.X_test.

        Args:
            method: str, one of ['drop', 'cap', 'impute']
            threshold_method: str, one of ['iqr', 'zscore']
        """
        import numpy as np

        X_train = self.X_train.copy()
        X_val = self.X_val.copy()
        X_test = self.X_test.copy()

        for col in X_train.select_dtypes(include=[np.number]).columns:
            if threshold_method == "":
                cur_threshold_method = self.get_threshold_method(X_train[col])
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
            self.logger.info(
                f"{len(outliers_train)} outliers found in {col}, threshold method: {cur_threshold_method}"
            )
            if method == "drop":
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

            elif method == "cap":
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

            elif method == "impute":
                median = X_train[col].median()
                X_train.loc[outliers_train, col] = median
                X_val.loc[
                    (X_val[col] < lower_bound) | (X_val[col] > upper_bound), col
                ] = median
                X_test.loc[
                    (X_test[col] < lower_bound) | (X_test[col] > upper_bound),
                    col,
                ] = median

            else:
                raise ValueError("Unsupported handling method")

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test

    def preprocess(self):
        self.logger.info("[MAGENTA]\nStarting preprocessing")
        result = self.eda.load_data()
        assert self.eda.df_train is not None
        if result != "":
            return result
        self.eda.set_column_types()

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

        # 4 handle outliers
        self.handle_outliers(method="cap", threshold_method="")

        # 5 missing values
        # 6 encoding
        # 7 normalizing/scaling
        # 8 dim. reduction
        self.logger.info("[MAGENTA] I am done here...")
        return "Done! So far...."
