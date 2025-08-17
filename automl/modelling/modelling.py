# internal libraries
from preprocessing import preprocess
from library import Logger
from .models import models
from .modelselection import run_kfold_evaluation

# external libraries

from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split


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
        # first make sure the target is available
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

        self.X_full, self.y_full, meta_data = preprocess(
            X=X_train, y=y_train, logger=self.logger
        )
        meta_data["target"] = self.target
        self.dataset_type: str = meta_data["dataset_type"]
        self.logger.warning(msg=f"{self.dataset_type} found!")
        self.logger.error(msg=f"{meta_data}")
        self.split_validation_data()
        results = run_kfold_evaluation(
            X=self.X_val_full,
            y=self.y_val_full,
            models=models,
            dataset_type=self.dataset_type,
            logger=self.logger,
        )
        self.logger.warning(msg=results)

    def split_validation_data(
        self,
        random_state=42,
        test_size=0.2,
        shuffle=True,
    ) -> None:
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
