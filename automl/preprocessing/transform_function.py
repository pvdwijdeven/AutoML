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

# self.target_transformer = self.step_outputs["standardize_target"][
#     "target_transformer"
# ]

# external imports
from typing import Optional, Dict, List, Any


class AutomlTransformer:

    def __init__(
        self,
        logger: Optional[Logger] = None,
    ) -> None:
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger: Logger = logger
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
                "config": {
                    "categorical_only": True,
                },
            },
            {
                "name": "drop_strings",
                "function": drop_strings,
            },
            {
                "name": "auto_encode_features",
                "function": auto_encode_features,
            },
            {
                "name": "handle_missing_values_num",
                "function": handle_missing_values,
                "config": {
                    "categorical_only": False,
                },
            },
            {
                "name": "handle_outliers_after",
                "function": handle_outliers,
                "config": {"before": False},
            },
            {
                "name": "normalize_columns",
                "function": normalize_columns,
            },
        ]

    def fit(self, X_train, y_train):
        self.step_outputs: Dict[str, Dict[str, Any]] = {}
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        for step in self.steps:
            self.logger.info(
                msg=f"[ORANGE]- fit/transform train: {step["name"]}"
            )
            X_train, y_train, step["params"] = step["function"](
                X_train,
                y_train,
                fit=True,
                step_params={},
                logger=self.logger,
                step_outputs=self.step_outputs.copy(),
                **step.get("config", {}),
            )

            self.step_outputs[step["name"]] = step["params"]

        self.logger.debug(msg=self.step_outputs)

    def transform(self, X_test):
        self.X_test = X_test.copy()

        for step in self.steps:
            self.logger.info(msg=f"[YELLOW]- transform test: {step["name"]}")
            self.X_test, self.y_test, _ = step["function"](
                X_test,
                None,
                fit=False,
                step_params=step["params"],
                logger=self.logger,
                step_outputs=self.step_outputs.copy(),
                **step.get("config", {}),
            )

        return self.X_test
