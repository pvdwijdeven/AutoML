# internal libraries
from preprocessing import AutoML_Preprocess
from library import Logger
from .models import models
from .scoring import (
    write_to_output,
    summarize_results,
    scoring_per_dataset_type,
    get_score,
)

# external libraries
from time import perf_counter
from typing import Dict, Optional, Any, List
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
import os


class AutoML_Modeling:

    def __init__(
        self,
        target: str,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        df_test: Optional[pd.DataFrame] = None,
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
        self.X_val_train: pd.DataFrame = X_train
        self.y_val_train: pd.Series = y_train
        self.output_file: str = output_file
        self.df_test = df_test
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger: Logger = logger
        self.dataset_type: str = self.detect_dataset_type(
            target=self.y_val_train
        )
        self.logger.warning(f"{self.dataset_type} found!")
        top_models = self.train_test_kfold_loop(
            dict_models=models[self.dataset_type],
            output_file=self.output_file.replace("result", "model_selection"),
        )
        self.hyper_top_models(top_models=top_models)
        # assert isinstance(model_name, str)
        # self.internal_test(model_name=model_name, model=model)
        # if self.df_test is not None:
        #     self.external_prediction(model_name=model_name, model=model)

    def hyper_top_models(self, top_models):

        hyper_models = {}
        _top_hyper_models = self.train_test_kfold_loop(
            dict_models=hyper_models,
            output_file=self.output_file.replace("result", "hyper"),
        )
        pass

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
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_type: str,
        model_dict: Dict[str, Any],
        save_file: str = "",
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
                    self.X_val_test.reset_index(drop=True),
                    pd.Series(y_test, name="y_test").reset_index(drop=True),
                    #     pd.Series(y_pred, name="y_pred").reset_index(drop=True),
                ],
                axis=1,
            )

            # Save to CSV (or any other format)
            df_out.to_csv(save_file, index=False)
        for name, model in model_dict.items():
            self.logger.info(msg=f"[GREEN]- Training on {name}")
            start_time = perf_counter()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            end_time = perf_counter()
            if self.target_transformer is not None:
                y_pred = self.target_transformer.inverse_transform(
                    y_scaled=y_pred
                )
            if save_file != "":
                df_out = pd.concat(
                    [
                        X_test.reset_index(drop=True),
                        pd.Series(y_test, name="y_test").reset_index(drop=True),
                        #     pd.Series(y_pred, name="y_pred").reset_index(drop=True),
                    ],
                    axis=1,
                )
            result = get_score(
                dataset_type=self.dataset_type,
                X_test=X_test,
                y_test=y_test,
                y_pred=y_pred,
                model=model,
                name=name,
                end_time=end_time,
                start_time=start_time,
            )
            results.setdefault(name, {}).update(result[name])
        return results

    def train_test_kfold_loop(
        self,
        dict_models,
        output_file,
        test_size=0.2,
        random_state=42,
        n_splits=5,
        shuffle=True,
    ) -> List[Dict[str, Any]]:
        """
        X_full/y_full: complete dataset
            - X_val_full/y_val_full: complete validation dataset (80% of X_full/y_full)
                - Kfold 5x:
                    - X_val_train/y_val_train: 80% of X_val_full
                    - X_val_test/y_val_test: 20% of X_val_full
            - X_final_test, y_final_test: complete test set against X_val_full (20% of X_full/y_full)
            
            X_val_train/y_val_train is used for selecting model and hypertuning it agains X_val_test and y_val_test
            After selection, the X_val_full/y_val_full is used to train that model and test against X_final_test/y_final_test\
        
        """
        self.X_full: pd.DataFrame = self.X_val_train.copy()
        self.y_full: pd.Series[Any] = self.y_val_train.copy()

        # First, split into train and test sets
        (
            self.X_val_full,
            self.X_final_test,
            self.y_val_full,
            self.y_final_test,
        ) = train_test_split(
            self.X_full,
            self.y_full,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        # Initialize KFold on training data only
        kf = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        total_result = {}
        # Loop over K folds
        for fold_idx, (train_index, val_index) in enumerate(
            kf.split(self.X_val_full)
        ):
            self.logger.info(
                f"[MAGENTA]Testing run {fold_idx + 1} of {n_splits}"
            )
            # Split train and validation sets for this fold
            self.X_val_train = self.X_val_full.iloc[train_index].reset_index(
                drop=True
            )
            self.y_val_train = self.y_val_full.iloc[train_index].reset_index(
                drop=True
            )
            self.X_val_test = self.X_val_full.iloc[val_index].reset_index(
                drop=True
            )
            self.y_val_test = self.y_val_full.iloc[val_index].reset_index(
                drop=True
            )
            cur_prepro = AutoML_Preprocess(
                X_train=self.X_val_train,
                y_train=self.y_val_train,
                X_test=self.X_val_test,
                y_test=self.y_val_test,
                logger=self.logger,
            )
            (
                self.X_val_train,
                self.y_val_train,
                self.X_val_test,
                self.y_val_test,
                self.target_transformer,
            ) = cur_prepro.preprocess()
            result: Dict[str, Dict[str, float]] = (
                self.train_and_evaluate_models(
                    X_train=self.X_val_train,
                    y_train=self.y_val_train,
                    X_test=self.X_val_test,
                    y_test=self.y_val_test,
                    dataset_type=self.dataset_type,
                    model_dict=dict_models,
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
        self.scoring = scoring_per_dataset_type[self.dataset_type]
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        summary_df, top_models = summarize_results(
            results_dict=total_result,
            model_dict=models[self.dataset_type],
            scoring=self.scoring,
        )
        df_top_models = pd.DataFrame(top_models)

        write_to_output(
            output_file=self.output_file,
            summary_df=summary_df.reset_index(),
            top_models=df_top_models,
        )
        return top_models

    def internal_test(self, model_name: str, model: Any):
        assert isinstance(self.y_val_full, pd.Series)
        cur_prepro = AutoML_Preprocess(
            X_train=self.X_val_full,
            y_train=self.y_val_full,
            X_test=self.X_final_test,
            y_test=self.y_final_test,
            logger=self.logger,
        )
        (
            self.X_val_full,
            self.y_val_full,
            self.X_final_test,
            self.y_final_test,
            self.target_transformer,
        ) = cur_prepro.preprocess()
        assert model is not None
        assert isinstance(model_name, str)
        result: Dict[str, Dict[str, float]] = self.train_and_evaluate_models(
            X_train=self.X_val_full,
            y_train=self.y_val_full,
            X_test=self.X_final_test,
            y_test=self.y_final_test,
            dataset_type=self.dataset_type,
            model_dict={model_name: model},
            save_file="",
        )
        self.logger.warning(msg=result[model_name][self.scoring])

    def external_prediction(self, model_name: str, model: Any):
        assert self.df_test is not None
        cur_prepro = AutoML_Preprocess(
            X_train=self.X_full,
            y_train=self.y_full,
            X_test=self.df_test,
            y_test=None,
            logger=self.logger,
        )
        (
            X,
            y,
            X_test,
            _y_test,
            self.target_transformer,
        ) = cur_prepro.preprocess()

        assert model is not None
        assert isinstance(model_name, str)
        model.fit(X, y)
        y_pred = model.predict(X_test)
        if self.target_transformer is not None:
            y_pred = self.target_transformer.inverse_transform(y_scaled=y_pred)
        # Check if self.y_full contains only True/False (case-insensitive)
        if pd.api.types.is_object_dtype(
            self.y_full
        ) or pd.api.types.is_bool_dtype(self.y_full):
            unique_vals = pd.Series(self.y_full).dropna().unique()
            unique_str = set(str(val).lower() for val in unique_vals)
            if unique_str.issubset({"true", "false"}):
                # Determine the original casing for True/False in y_full
                true_val = next(
                    val for val in unique_vals if str(val).lower() == "true"
                )
                false_val = next(
                    val for val in unique_vals if str(val).lower() == "false"
                )
                # Map y_pred 1/0 to the same casing as in y_full
                y_pred = [
                    true_val if pred == 1 else false_val for pred in y_pred
                ]
        # Combine the first column of self.df_test and y_pred into a new DataFrame
        try:
            output_df = pd.DataFrame(
                {
                    self.df_test.columns[0]: self.df_test.iloc[:, 0].values,
                    self.target: y_pred,
                }
            )
        except ValueError:
            self.logger.error(
                f"Size of y_pred: {len(y_pred)}, df_test: {self.df_test.shape}"
            )
            exit()
        # Save as CSV
        my_file = self.output_file.replace("html", "csv").replace(
            "result", "submission"
        )
        # TODO get target back in original form
        output_df.to_csv(my_file, index=False)
