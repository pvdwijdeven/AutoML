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
from .scoring import (
    select_top_models,
    write_to_output,
    get_scoring,
    lrmse_scorer,
    flexible_scorer,
)
from .hypertuning import param_grids, param_grids_detailed

# external libraries

from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os


class AutomlModeling:

    def __init__(
        self,
        target: str,
        scoring: str,
        X_original: pd.DataFrame,
        y_original: Optional[pd.Series] = None,
        df_test: Optional[pd.DataFrame] = None,
        output_file: str = "",
        logger: Optional[Logger] = None,
    ) -> None:
        # first make sure the target is available
        if y_original is None:
            if target not in X_original.columns:
                raise ValueError(
                    f"Target column '{target}' not found in X_train"
                )
            y_original = X_original[target]
            X_original = X_original.drop(target, axis=1)
        self.X_original = X_original
        self.y_original = y_original
        self.target: str = target
        self.output_file: str = output_file
        self.df_test = df_test
        self.scoring = scoring
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger: Logger = logger
        self.start_modelling()

    def start_modelling(self):
        self.split_validation_data()

        self.X_val_prepro, self.y_val_prepro, self.meta_data = preprocess(
            X=self.X_val, y=self.y_val, logger=self.logger
        )
        self.meta_data["target"] = self.target
        self.dataset_type: str = self.meta_data["dataset_type"]
        self.scoring = get_scoring(self.scoring, self.dataset_type)
        if self.scoring == "lrmse":
            self.scorer = lrmse_scorer
        else:
            self.scorer = self.scoring
        self.meta_data["scoring"] = self.scoring
        self.meta_data["scorer"] = self.scorer
        self.logger.info(msg=f"[BLUE] Dataset type: {self.dataset_type}")

        self.logger.info(msg="[MAGENTA] Starting primarily model selction")

        # Step 1 - model selection 10 models, 1 parameter per set
        checkpoint = self.load_step(step=1)
        if checkpoint is None:
            results = run_kfold_evaluation(
                X=self.X_val_prepro,
                y=self.y_val_prepro,
                models=models,
                dataset_type=self.dataset_type,
                logger=self.logger,
                scoring=self.scorer,
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
            self.save_step(
                step=1, obj=(df_results, top_selection, self.meta_data)
            )
        else:
            df_results, top_selection, self.meta_data = checkpoint

        self.logger.info(
            msg=f"[MAGENTA] Starting hypertuning top X model selction ({df_results})"
        )
        # Step 2 model selection top (max) 5 of step 1, small hypertuning grid set

        checkpoint = self.load_step(2)
        if checkpoint is None:
            best_grid_model = run_kfold_grid_search(
                dataset_type=self.dataset_type,
                scoring=self.scorer,
                top_models=top_selection,
                param_grids=param_grids,
                X=self.X_val_prepro,
                y=self.y_val_prepro,
                logger=self.logger,
            )
            self.save_step(step=2, obj=(best_grid_model, self.meta_data))
        else:
            best_grid_model, self.meta_data = checkpoint

        best_model_name, best_score = get_best_model_name(
            results=best_grid_model
        )
        self.logger.info(
            msg=f"[MAGENTA]Best model: {best_model_name}, score: {best_score}"
        )

        # Step 3 hyperparameter tuning on best model of step 2 is done
        checkpoint = self.load_step(step=3)
        if checkpoint is None:
            fine_tuned_model = run_kfold_grid_search(
                dataset_type=self.dataset_type,
                scoring=self.scorer,
                top_models=[{"model_name": best_model_name}],
                param_grids=param_grids_detailed,
                X=self.X_val_prepro,
                y=self.y_val_prepro,
                logger=self.logger,
            )
            self.save_step(step=3, obj=(fine_tuned_model, self.meta_data))
        else:
            fine_tuned_model, self.meta_data = checkpoint

        write_to_output(
            output_file=self.output_file,
            summary_df=df_results,
            top_models=pd.DataFrame(data=top_selection),
            best_grid=create_results_html_table(best_grid_model),
            final_result=create_results_html_table(fine_tuned_model),
        )
        best_model_name, best_score = get_best_model_name(
            results=fine_tuned_model
        )
        self.logger.info(
            msg=f"Best model: {best_model_name}, score: {best_score}"
        )
        # Step 4 - final test on unexposed test set on best model of step 3.

        if "encode_target" in self.meta_data:
            self.y_test_prepro = self.meta_data["encode_target"][
                "encoder"
            ].transform(y=self.y_final_test)
        elif "standardize_target" in self.meta_data:
            self.y_test_prepro = self.meta_data["standardize_target"][
                "target_transformer"
            ].transform(y=self.y_final_test)
        checkpoint = self.load_step(step=4)
        if checkpoint is None:
            best_model = next(iter(fine_tuned_model.values()))["best_estimator"]
            best_model.fit(self.X_val_prepro, self.y_val_prepro)
            score = flexible_scorer(
                estimator=best_model,
                X=self.X_final_test,
                y=self.y_test_prepro,
                scorer_param=self.scorer,
            )
            self.save_step(step=4, obj=(best_model, score, self.meta_data))
        else:
            best_model, score, self.meta_data = checkpoint

        self.logger.info(
            msg=f"[ORANGE] Scoring on 20% untouched dataset: {score}"
        )

        # final step if df_test is available
        if self.df_test is not None:
            self.X_full_prepro, self.y_full_prepro, self.meta_data = preprocess(
                X=self.X_original, y=self.y_original, logger=self.logger
            )
            best_model.fit(self.X_full_prepro, self.y_full_prepro)
            self.y_pred = best_model.predict(self.df_test)
            if "encode_target" in self.meta_data:
                self.y_pred = self.meta_data["encode_target"][
                    "encoder"
                ].inverse_transform(y=self.y_pred)
            elif "standardize_target" in self.meta_data:
                self.y_pred = self.meta_data["standardize_target"][
                    "target_transformer"
                ].inverse_transform(y=self.y_pred)
            first_col = self.df_test.iloc[:, 0]
            # Create a DataFrame combining the first column and y_pred
            df = pd.DataFrame(
                data={first_col.name: first_col, self.target: self.y_pred}
            )

            # Save DataFrame to CSV
            df.to_csv(
                self.output_file.replace("results.html", "submission.csv"),
                index=False,
            )
            self.logger.info(
                f"Submissing saved to {self.output_file.replace("results.html", "submission.csv")}"
            )
        self.logger.info(msg="[MAGENTA] DONE")

    def split_validation_data(
        self,
        random_state=42,
        test_size=0.2,
        shuffle=True,
    ) -> None:
        (
            self.X_val,
            self.X_final_test,
            self.y_val,
            self.y_final_test,
        ) = train_test_split(
            self.X_original,
            self.y_original,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

    def save_step(self, step: int, obj):

        path = self.output_file.replace("results.html", f"step{step}.pkl")
        with open(file=path, mode="wb") as f:
            pickle.dump(obj=obj, file=f)
        self.logger.info(msg=f"[GREEN] Saved checkpoint: {path}")

    def load_step(self, step: int):
        path = self.output_file.replace("results.html", f"step{step}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.logger.info(f"[CYAN] Loaded checkpoint: {path}")
                return pickle.load(f)
        return None
