# internal libraries
from preprocessing import AutoML_Preprocess
from library import Logger
from .models import models
from .scoring import write_to_output

# external libraries
from time import perf_counter
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, KFold
from math import sqrt
import os


class AutoML_Modeling:

    def __init__(
        self,
        target: str,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        output_file: str = "",
        logger: Optional[Logger] = None,
    ) -> None:
        if y_train is None:
            if target not in X_train.columns:
                raise ValueError(
                    f"Target column '{target}' not found in X_train"
                )
            y_train = X_train[target]
            X_train = X_train.drop(target, axis=1)

        self.target: str = target
        self.X_train: pd.DataFrame = X_train
        self.y_train: pd.Series = y_train
        self.output_file: str = output_file
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger: Logger = logger
        self.dataset_type: str = self.detect_dataset_type(target=self.y_train)
        self.logger.warning(f"{self.dataset_type} found!")
        self.train_test_kfold_loop()

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
                target_df = pd.Series(data=target)
            elif target.ndim == 2:
                target_df = pd.DataFrame(data=target)
            else:
                raise ValueError("Target must be 1D or 2D array-like")

        # Determine if multi-label (2D target)
        if isinstance(target_df, pd.DataFrame) and target_df.shape[1] > 1:
            # Check if binary indicators (0/1) in multiple columns -> multi-label classification
            unique_vals = pd.unique(values=target_df.values.ravel())
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
        is_numeric: bool = pd.api.types.is_numeric_dtype(
            arr_or_dtype=target_series
        )

        # Get unique values
        unique_vals = target_series.dropna().unique()
        n_unique: int = len(unique_vals)

        # Heuristic: if numeric with many unique values -> regression
        if is_numeric and n_unique > 20:
            return "regression"

        # Check if target is categorical (string or int categories)
        # If only two unique classes
        if n_unique == 2:
            # Calculate the class distribution ratio
            counts: pd.Series[float] = target_series.value_counts(
                normalize=True
            )

            # Define threshold for "high imbalance" (e.g. minority class <= 5%)
            imbalance_threshold = 0.05

            # Check minority class proportion
            minority_class_ratio: float = counts.min()

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
            unique_vals_sorted = np.sort(a=unique_vals)
            diffs = np.diff(a=unique_vals_sorted)
            # Check if differences are all 1 and values are integers => ordinal (e.g. ratings)
            if np.all(a=diffs == 1) and np.all(
                a=unique_vals_sorted == unique_vals_sorted.astype(dtype=int)
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
        self, dataset_type: str, save_file: str = ""
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
        if save_file != "":
            df_out = pd.concat(
                [
                    self.X_test.reset_index(drop=True),
                    pd.Series(self.y_test, name="y_test").reset_index(
                        drop=True
                    ),
                    #     pd.Series(y_pred, name="y_pred").reset_index(drop=True),
                ],
                axis=1,
            )

            # Save to CSV (or any other format)
            df_out.to_csv(save_file, index=False)
        for name, model in models[dataset_type].items():
            self.logger.info(msg=f"[GREEN]- Training on {name}")
            start_time = perf_counter()
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            end_time = perf_counter()
            if self.target_transformer is not None:
                y_pred = self.target_transformer.inverse_transform(
                    y_scaled=y_pred
                )
            if save_file != "":
                df_out = pd.concat(
                    [
                        self.X_test.reset_index(drop=True),
                        pd.Series(self.y_test, name="y_test").reset_index(
                            drop=True
                        ),
                        #     pd.Series(y_pred, name="y_pred").reset_index(drop=True),
                    ],
                    axis=1,
                )
            # Choose metric depending on dataset type
            if dataset_type in (
                "binary_classification",
                "imbalanced_binary_classification",
            ):
                # You can choose metrics as needed; here we use accuracy and F1 + AUC if possible
                acc = accuracy_score(y_true=self.y_test, y_pred=y_pred)
                f1 = f1_score(y_true=self.y_test, y_pred=y_pred)

                # Try to get predicted probabilities for AUC if supported
                try:
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                    auc = roc_auc_score(y_true=self.y_test, y_score=y_proba)
                except ValueError:
                    auc = None

                results[name] = {
                    "accuracy": acc,
                    "f1_score": f1,
                    "roc_auc": auc,
                    "time": end_time - start_time,
                }

            elif dataset_type == "multi_class_classification":
                acc = accuracy_score(y_true=self.y_test, y_pred=y_pred)
                f1 = f1_score(
                    y_true=self.y_test, y_pred=y_pred, average="weighted"
                )
                results[name] = {
                    "accuracy": acc,
                    "f1_score_weighted": f1,
                    "time": end_time - start_time,
                }

            elif dataset_type == "multi_label_classification":
                # For multi-label, use an appropriate metric such as average F1 per label
                f1_macro = f1_score(
                    y_true=self.y_test, y_pred=y_pred, average="macro"
                )
                f1_micro = f1_score(
                    y_true=self.y_test, y_pred=y_pred, average="micro"
                )
                results[name] = {
                    "f1_macro": f1_macro,
                    "f1_micro": f1_micro,
                    "time": end_time - start_time,
                }

            elif dataset_type == "ordinal_regression":
                # Could treat as regression or classification; here treat as classification
                acc = accuracy_score(y_true=self.y_test, y_pred=y_pred)
                results[name] = {
                    "accuracy": acc,
                    "time": end_time - start_time,
                }

            elif dataset_type == "regression":
                epsilon = (
                    1e-15  # small number to avoid log(0), change if needed
                )
                lrmse: float = sqrt(
                    mean_squared_error(
                        y_true=np.log(self.y_test + epsilon),
                        y_pred=np.log(np.maximum(y_pred, epsilon)),
                    ),
                )

                mse: float = mean_squared_error(
                    y_true=self.y_test, y_pred=y_pred
                )
                rmse: float = sqrt(mse)
                r2: float = r2_score(y_true=self.y_test, y_pred=y_pred)
                results[name] = {
                    "lrmse": lrmse,
                    "rmse": rmse,
                    "mse": mse,
                    "r2": r2,
                    "time": end_time - start_time,
                }

            else:
                # Fallback metric if unknown
                mse = mean_squared_error(y_true=self.y_test, y_pred=y_pred)
                results[name] = {
                    "mse": mse,
                    "time": end_time - start_time,
                }

        return results

    def train_test_kfold_loop(
        self, test_size=0.2, random_state=42, n_splits=2, shuffle=True
    ):

        self.X_train_full: pd.DataFrame = self.X_train.copy()
        self.y_train_full: pd.Series[Any] = self.y_train.copy()

        # First, split into train and test sets
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = (
            train_test_split(
                self.X_train_full,
                self.y_train_full,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
            )
        )

        # Initialize KFold on training data only
        kf = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        total_result = {}
        # Loop over K folds
        for fold_idx, (train_index, val_index) in enumerate(
            kf.split(self.X_train_val)
        ):
            self.logger.info(
                f"[MAGENTA]Testing run {fold_idx + 1} of {n_splits}"
            )
            # Split train and validation sets for this fold
            self.X_train = self.X_train_val.iloc[train_index].reset_index(
                drop=True
            )
            self.y_train = self.y_train_val.iloc[train_index].reset_index(
                drop=True
            )
            self.X_test = self.X_train_val.iloc[val_index].reset_index(
                drop=True
            )
            self.y_test = self.y_train_val.iloc[val_index].reset_index(
                drop=True
            )
            cur_prepro = AutoML_Preprocess(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
                logger=self.logger,
            )
            (
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.target_transformer,
            ) = cur_prepro.preprocess()
            result: Dict[str, Dict[str, float]] = (
                self.train_and_evaluate_models(
                    dataset_type=self.dataset_type,
                    save_file=self.output_file.replace("html", "csv")
                    .replace(
                        "result",
                        f"fold{fold_idx}",
                    )
                    .replace("export", "export/csv"),
                )
            )
            total_result[fold_idx] = result
        self.logger.debug(msg=total_result)
        self.logger.info(
            msg=f"[GREEN]- Writing report file to {self.output_file}"
        )
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        write_to_output(self.output_file, total_result)
