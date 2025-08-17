# internal imports
from library import Logger
from .general import (
    drop_duplicate_columns,
    drop_constant_columns,
    drop_strings,
)
from .outliers import (
    outlier_imputation_order,
    handle_outliers,
)
from .encoding import auto_encode_features
from .missing import handle_missing_values
from .standardizing import (
    normalize_columns,
)

# external imports
from typing import Optional, Dict, List, Any, Self
import pandas as pd


class AutomlTransformer:
    """
    Automated data preprocessing and transformation pipeline for machine learning.

    This class performs a sequence of data preprocessing steps on a given dataset,
    storing transformation parameters during the fitting phase and applying them
    consistently during transformation.

    Steps include:
        - Dropping duplicate columns
        - Dropping constant columns
        - Handling missing values (categorical and numerical)
        - Encoding categorical features
        - Outlier imputation and handling (before and after missing value imputation)
        - Dropping string columns
        - Normalizing column values

    Attributes
    ----------
    logger : Logger
        Custom logger for tracking pipeline execution.
    steps : List[Dict[str, Any]]
        Ordered list of processing steps with functions and configurations.
    meta_data : Dict[str, Dict[str, Any]]
        Stores fitted parameters for each step to ensure consistency during transformation.
    X_train, y_train, X_test, y_test : Any
        Copies of training/testing datasets.
    """

    def __init__(self, logger: Optional[Logger] = None) -> None:
        """
        Initialize the AutomlTransformer pipeline.

        Parameters
        ----------
        logger : Optional[Logger]
            Custom logger for pipeline. If not provided, a default Logger is created.
        """
        if logger is None:
            self._init_logger_filename = ""
        else:
            self._init_logger_filename = logger.filename
        self.logger = logger or Logger(
            level_console=Logger.INFO,
            level_file=Logger.DEBUG,
            filename="",
            wx_handler=None,
        )

        # Define the pipeline steps with optional config parameters
        self.steps: List[Dict[str, Any]] = [
            {
                "name": "drop_duplicate_columns",
                "function": drop_duplicate_columns,
            },
            {
                "name": "drop_constant_columns",
                "function": drop_constant_columns,
            },
            {
                "name": "outlier_imputation_order",
                "function": outlier_imputation_order,
            },
            {
                "name": "handle_outliers_before",
                "function": handle_outliers,
                "config": {"before": True},
            },
            {
                "name": "handle_missing_values_cat",
                "function": handle_missing_values,
                "config": {"categorical_only": True},
            },
            {"name": "drop_strings", "function": drop_strings},
            {"name": "auto_encode_features", "function": auto_encode_features},
            {
                "name": "handle_missing_values_num",
                "function": handle_missing_values,
                "config": {"categorical_only": False},
            },
            {
                "name": "handle_outliers_after",
                "function": handle_outliers,
                "config": {"before": False},
            },
            {"name": "normalize_columns", "function": normalize_columns},
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the logger instance to avoid pickling errors
        if "logger" in state:
            del state["logger"]
        return state

    def __setstate__(self, state):
        # Restore all other attributes
        self.__dict__.update(state)
        # Recreate logger using the original init parameter if any
        self.logger = Logger(
            level_console=Logger.INFO,
            level_file=Logger.DEBUG,
            filename=self._init_logger_filename,
            wx_handler=None,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Self:
        """
        Fit the transformation pipeline to the training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature dataset.
        y_train : pd.Series
            Training target dataset.

        Notes
        -----
        Applies each step in sequence, fitting step parameters which are stored
        for later use during transformation.
        """
        self.meta_data: Dict[str, Dict[str, Any]] = {}
        self.X_train, self.y_train = X_train.copy(), y_train.copy()

        for step in self.steps:

            _dummy_X, _dummy_y, step_params = step["function"](
                self.X_train,
                self.y_train,
                fit=True,
                step_params={},
                logger=self.logger,
                meta_data=self.meta_data.copy(),
                **step.get("config", {}),
            )

            step["params"] = step_params
            self.meta_data[step["name"]] = step_params

            self.X_train, _dummy, _ = step["function"](
                self.X_train,
                None,
                fit=False,
                step_params=step.get("params", {}),
                logger=self.logger,
                meta_data=self.meta_data.copy(),
                **step.get("config", {}),
            )

            # store fitted step parameters (metadata)

        self.logger.debug(msg=f"Fitted metadata: {self.meta_data}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the test data using the fitted pipeline.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature dataset to be transformed.

        Returns
        -------
        pd.DataFrame
            Transformed test features.
        """

        self.X = X.copy()
        self.y = None

        for step in self.steps:
            self.X, self.y, _ = step["function"](
                self.X,
                None,
                fit=False,
                step_params=step.get("params", {}),
                logger=self.logger,
                meta_data=self.meta_data.copy(),
                **step.get("config", {}),
            )

        return self.X
