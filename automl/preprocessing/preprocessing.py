# internal imports
from library import Logger
from .general import (
    drop_duplicate_rows,
    drop_duplicate_columns,
    drop_constant_columns,
    drop_strings,
)
from .outliers import (
    skip_outliers,
    outlier_imputation_order,
    handle_outliers,
)
from .encoding import auto_encode_features, encode_target
from .missing import handle_missing_values_cat, handle_missing_values_num
from .standardizing import normalize_columns

# external imports
import pandas as pd
from typing import Optional, Dict, List, Any


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
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        logger: Optional[Logger] = None,
    ) -> None:
        self.X_train: pd.DataFrame = X_train
        self.y_train: pd.Series = y_train
        self.X_test: pd.DataFrame = X_test
        self.y_test: Optional[pd.Series] = y_test
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger: Logger = logger

    def preprocess(
        self,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series | None]:

        self.steps: List[Dict[str, Any]] = [
            {
                "name": "drop_duplicate_rows",
                "params": None,
                "function": drop_duplicate_rows,
                "target_aware": True,
            },
            {
                "name": "drop_duplicate_columns",
                "params": None,
                "function": drop_duplicate_columns,
                "target_aware": False,
            },
            {
                "name": "drop_constant_columns",
                "params": None,
                "function": drop_constant_columns,
                "target_aware": False,
            },
            {
                "name": "skip_outliers",
                "params": None,
                "function": skip_outliers,
                "target_aware": True,
            },
            {
                "name": "outlier_imputation_order",
                "params": None,
                "function": outlier_imputation_order,
                "target_aware": True,
            },
            {
                "name": "handle_outliers_before",
                "params": None,
                "function": handle_outliers,
                "target_aware": True,
                "config": {"before": True},
            },
            {
                "name": "handle_missing_values_cat",
                "params": None,
                "function": handle_missing_values_cat,
                "target_aware": True,
                "config": {
                    "categorical_only": True,
                },
            },
            {
                "name": "encode_target",
                "params": None,
                "function": encode_target,
                "target_aware": True,
            },
            {
                "name": "drop_strings",
                "params": None,
                "function": drop_strings,
                "target_aware": True,
            },
            {
                "name": "auto_encode_features",
                "params": None,
                "function": auto_encode_features,
                "target_aware": False,
            },
            {
                "name": "handle_missing_values_num",
                "params": None,
                "function": handle_missing_values_num,
                "target_aware": True,
                "config": {
                    "categorical_only": False,
                },
            },
            {
                "name": "handle_outliers_after",
                "params": None,
                "function": handle_outliers,
                "target_aware": True,
                "config": {"before": False},
            },
            {
                "name": "normalize_columns",
                "params": None,
                "function": normalize_columns,
                "target_aware": True,
            },
        ]

        self.step_outputs: Dict[str, Dict[str, Any]] = {}
        for step in self.steps:
            self.logger.info(msg=f"[ORANGE]- training/fitting: {step["name"]}")
            self.X_train, self.y_train, step["params"] = step["function"](
                self.X_train,
                self.y_train,
                fit=True,
                step_params={},
                target_aware=step["target_aware"],
                logger=self.logger,
                step_outputs=self.step_outputs.copy(),
                **step.get("config", {}),
            )

            self.step_outputs[step["name"]] = step["params"]

        self.logger.debug(msg=self.step_outputs)

        for step in self.steps:
            self.logger.info(msg=f"[YELLOW]- transforming: {step["name"]}")
            self.X_test, self.y_test, _ = step["function"](
                self.X_test,
                self.y_test,
                fit=False,
                step_params=step["params"],
                target_aware=step["target_aware"],
                logger=self.logger,
                step_outputs=self.step_outputs.copy(),
                **step.get("config", {}),
            )
        return self.X_train, self.y_train, self.X_test, self.y_test
