# internal libraries
from preprocessing import preprocess
from library import Logger
from .models import models
from .modelselection import (
    run_kfold_evaluation,
    run_kfold_grid_search,
    create_results_html_table,
    get_best_model_name,
)
from .scoring import select_top_models, write_to_output
from .hypertuning import param_grids, param_grids_detailed

# external libraries

from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split


class AutomlModeling:

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
        self.logger.info(msg=f"[BLUE] Dataset type: {self.dataset_type}")
        self.split_validation_data()
        self.logger.info(msg="[MAGENTA] Starting primarily model selction")
        results = run_kfold_evaluation(
            X=self.X_val_full,
            y=self.y_val_full,
            models=models,
            dataset_type=self.dataset_type,
            logger=self.logger,
        )
        df_results = pd.DataFrame(
            [
                {
                    "model": model,
                    "mean_score": details["mean_score"],
                    "std_score": details["std_score"],
                }
                for model, details in results.items()
            ]
        )
        top_selection = select_top_models(summary_df=df_results)
        self.logger.info(
            msg="[MAGENTA] Starting hypertuning top X model selction"
        )
        best_grid_model = run_kfold_grid_search(
            dataset_type=self.dataset_type,
            top_models=top_selection,
            param_grids=param_grids,
            X=self.X_val_full,
            y=self.y_val_full,
            logger=self.logger,
        )
        best_model, best_score = get_best_model_name(results=best_grid_model)
        self.logger.info(msg=f"Best model: {best_model}, score: {best_score}")
        final_result = run_kfold_grid_search(
            dataset_type=self.dataset_type,
            top_models=[{"model_name": best_model}],
            param_grids=param_grids_detailed,
            X=self.X_val_full,
            y=self.y_val_full,
            logger=self.logger,
        )

        write_to_output(
            output_file=self.output_file,
            summary_df=df_results,
            top_models=pd.DataFrame(data=top_selection),
            best_grid=create_results_html_table(best_grid_model),
            final_result=create_results_html_table(final_result),
        )
        best_model, best_score = get_best_model_name(results=final_result)
        self.logger.info(msg=f"Best model: {best_model}, score: {best_score}")
        self.logger.info(msg="[MAGENTA] DONE")

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
