# Standard library imports
from typing import Any, Optional, Self

# Third-party imports
import pandas as pd
from pandas import DataFrame, Series

# Local application imports
from _automl_old.library import Logger

from .column_handling import (
    drop_duplicate_columns,
    drop_constant_columns,
    general_info,
    drop_strings,
)
from .outliers import handle_outliers
from .missing_values import handle_missing_values
from .encoding import auto_encode_features
from .standardizing import normalize_columns

class AutomlTransformer:
    """
    Automated data preprocessing and transformation pipeline for machine learning.

    This class performs a sequence of data preprocessing steps on a given dataset,
    storing transformation parameters during the fitting phase and applying them
    consistently during transformation. It follows a scikit-learn-like API.

    Steps include:
        - Dropping duplicate columns
        - Dropping constant columns
        - Outlier handling (pre-imputation)
        - Handling missing values (categorical and numerical)
        - Encoding categorical features
        - Outlier handling (post-imputation)
        - Dropping string columns
        - Normalizing column values

    Attributes
    ----------
    logger : Logger
        Custom logger for tracking pipeline execution.
    steps : list[dict[str, Any]]
        Ordered list of processing steps with functions and configurations.
    meta_data : dict[str, dict[str, Any]]
        Stores fitted parameters for each step (e.g., imputation means, dropped columns,
        normalization stats) to ensure consistency during transformation.
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
            # Store necessary info to recreate logger after pickling
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
        self.steps: list[dict[str, Any]] = [
            {
                "name": "drop_duplicate_columns",
                "function": drop_duplicate_columns,
            },
            {
                "name": "drop_constant_columns",
                "function": drop_constant_columns,
            },
            {
                "name": "general_info",
                "function": general_info,
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
            {
                "name": "encode_categorical_features",
                "function": auto_encode_features,
                "config": {"OHE_CARDINALITY_THRESHOLD": 10},
            },
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

        # Initialize meta_data to be populated during fit
        self.meta_data: dict[str, dict[str, Any]] = {}
        # Attributes to store transformed training data (optional, but useful for debugging)
        self.X_train_transformed: Optional[DataFrame] = None
        self.y_train_transformed: Optional[Series] = None

    def __getstate__(self):
        """Prepares the object for pickling by removing the logger."""
        state = self.__dict__.copy()
        # Remove the logger instance to avoid pickling errors
        if "logger" in state:
            del state["logger"]
        return state

    def __setstate__(self, state):
        """Restores the object after unpickling and recreates the logger."""
        # Restore all other attributes
        self.__dict__.update(state)
        # Recreate logger using the original init parameter
        self.logger = Logger(
            level_console=Logger.INFO,
            level_file=Logger.DEBUG,
            filename=self._init_logger_filename,
            wx_handler=None,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Self:
        """
        Fit the transformation pipeline to the training data.

        Applies each step in sequence, fitting step parameters and immediately
        transforming the data before passing it to the next step.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature dataset.
        y_train : pd.Series
            Training target dataset.

        Returns
        -------
        Self
            The fitted transformer instance.
        """
        self.meta_data = {}
        X, y = X_train.copy(), y_train.copy()

        for step in self.steps:
            step_name = step["name"]
            self.logger.info(f"Fitting and applying step: {step_name}")

            # FIT and TRANSFORM the data for the current step
            # The function is called once with fit=True. It must determine parameters
            # and apply the transformation to X and y in one go.
            X, y, step_params = step["function"](
                X,
                y,
                fit=True,
                step_params={},  # Step params are calculated here
                logger=self.logger,
                **step.get("config", {}),
            )

            # Store the fitted parameters (metadata)
            self.meta_data[step_name] = step_params

            # Store parameters within the step definition for the transform method
            step["params"] = step_params

        self.X_train_transformed = X
        self.y_train_transformed = y
        self.logger.debug(msg=f"Fitted metadata: {self.meta_data}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the feature data using the fitted pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataset to be transformed (e.g., test data or new data).

        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        X_transformed = X.copy()
        y_dummy = None  # We do not transform the target in the transform phase

        for step in self.steps:
            step_name = step["name"]
            self.logger.info(f"Applying transformation step: {step_name}")

            # Apply the transformation using the stored parameters
            X_transformed, y_dummy, _ = step["function"](
                X_transformed,
                y_dummy,
                fit=False,
                step_params=step.get("params", {}),  # Use stored parameters
                logger=self.logger,
                **step.get("config", {}),
            )

        # Optional: Store final size for reporting purposes
        self.meta_data["sizes"] = {"after_preprocessing": X_transformed.shape}
        return X_transformed
