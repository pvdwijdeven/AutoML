# internal libraries
from preprocessing import preprocess
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

        self.X_val_train, self.y_val_train, meta_data = preprocess(
            X=X_train, y=y_train, logger=self.logger
        )
        self.dataset_type: str = str(meta_data["dataset_type"])
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
