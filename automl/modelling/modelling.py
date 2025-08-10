# internal libraries
from library import Logger
from .models import models

# external libraries
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)


# For regression, populate models dict with regressors (LinearRegression, RFRegressor, etc)


class AutoML_Modeling:
    def __init__(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        df_test,
        logger: Logger | None = None,
    ) -> None:
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.df_test = df_test
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger = logger
        self.dataset_type = self.detect_dataset_type(self.y_train)
        result = self.train_and_evaluate_models(
            dataset_type=self.dataset_type,
        )
        self.logger.debug(result)

    def detect_dataset_type(
        self, target: pd.DataFrame | pd.Series | np.ndarray
    ) -> str:
        """
        Detect dataset type based on the target variable.

        Parameters:
        - target: array-like (1D or 2D), target values for supervised learning.

        Returns:
        - str: dataset type among:
        'binary_classification',
        'multi_class_classification',
        'multi_label_classification',
        'ordinal_regression',
        'regression'.
        """

        # Convert to pandas object for convenience
        if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
            target_df = target
        else:
            # If numpy array or list
            if target.ndim == 1:
                target_df = pd.Series(target)
            elif target.ndim == 2:
                target_df = pd.DataFrame(target)
            else:
                raise ValueError("Target must be 1D or 2D array-like")

        # Determine if multi-label (2D target)
        if isinstance(target_df, pd.DataFrame) and target_df.shape[1] > 1:
            # Check if binary indicators (0/1) in multiple columns -> multi-label classification
            unique_vals = pd.unique(target_df.values.ravel())
            if set(unique_vals).issubset({0, 1}):
                return "multi_label_classification"
            else:
                # Otherwise it's more complicated but treat as multi-label
                return "multi_label_classification"

        # For 1D target (Series)
        target_series = (
            target_df
            if isinstance(target_df, pd.Series)
            else target_df.iloc[:, 0]
        )

        # Check if target is numeric
        is_numeric = pd.api.types.is_numeric_dtype(target_series)

        # Get unique values
        unique_vals = target_series.dropna().unique()
        n_unique = len(unique_vals)

        # Heuristic: if numeric with many unique values -> regression
        if is_numeric and n_unique > 20:
            return "regression"

        # Check if target is categorical (string or int categories)
        # If only two unique classes
        if n_unique == 2:
            # Calculate the class distribution ratio
            counts = target_series.value_counts(normalize=True)

            # Define threshold for "high imbalance" (e.g. minority class <= 5%)
            imbalance_threshold = 0.05

            # Check minority class proportion
            minority_class_ratio = counts.min()

            if minority_class_ratio <= imbalance_threshold:
                return "imbalanced_binary_classification"
            else:
                return "binary_classification"

        # More than two unique classes:
        # If target is categorical or int but with ordered discrete small values,
        # attempt to detect ordinal by checking if sorted unique_vals are numeric and contiguous
        if not is_numeric:
            # Non-numeric classes likely multi-class
            return "multi_class_classification"
        else:
            # Numeric case:
            unique_vals_sorted = np.sort(unique_vals)
            diffs = np.diff(unique_vals_sorted)
            # Check if differences are all 1 and values are integers => ordinal (e.g. ratings)
            if np.all(diffs == 1) and np.all(
                unique_vals_sorted == unique_vals_sorted.astype(int)
            ):
                # Assume ordinal regression if unique count between 3 and 20
                if 3 <= n_unique <= 20:
                    return "ordinal_regression"
                else:
                    return "multi_class_classification"
            else:
                # Otherwise treat as multi-class classification if discrete numeric classes
                if n_unique <= 20:
                    return "multi_class_classification"
                else:
                    # If numeric with many unique classes but less than 20 (caught above regression >20)
                    return "multi_class_classification"

        # Fallback (should not get here)
        return "unknown"

    def train_and_evaluate_models(
        self, dataset_type: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate a dictionary of models.

        Params:
        - dataset_type: str identifying dataset type, e.g. "binary_classification", "regression", etc.

        Returns:
        - results: dict with model names as keys and evaluation metric(s) as values
        """

        results = {}
        if dataset_type not in models:
            return {}

        for name, model in models[dataset_type].items():
            self.logger.info(f"[GREEN]- Training on {name}")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.logger.info(f"[GREEN]- Scoring on {name}")
            # Choose metric depending on dataset type
            if dataset_type in (
                "binary_classification",
                "imbalanced_binary_classification",
            ):
                # You can choose metrics as needed; here we use accuracy and F1 + AUC if possible
                acc = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)

                # Try to get predicted probabilities for AUC if supported
                try:
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                    auc = roc_auc_score(self.y_test, y_proba)
                except ValueError:
                    auc = None

                results[name] = {
                    "accuracy": acc,
                    "f1_score": f1,
                    "roc_auc": auc,
                }

            elif dataset_type == "multi_class_classification":
                acc = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average="weighted")
                results[name] = {"accuracy": acc, "f1_score_weighted": f1}

            elif dataset_type == "multi_label_classification":
                # For multi-label, use an appropriate metric such as average F1 per label
                f1_macro = f1_score(self.y_test, y_pred, average="macro")
                f1_micro = f1_score(self.y_test, y_pred, average="micro")
                results[name] = {"f1_macro": f1_macro, "f1_micro": f1_micro}

            elif dataset_type == "ordinal_regression":
                # Could treat as regression or classification; here treat as classification
                acc = accuracy_score(self.y_test, y_pred)
                results[name] = {"accuracy": acc}

            elif dataset_type == "regression":
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                results[name] = {"mse": mse, "r2": r2}

            else:
                # Fallback metric if unknown
                mse = mean_squared_error(self.y_test, y_pred)
                results[name] = {"mse": mse}

        return results
