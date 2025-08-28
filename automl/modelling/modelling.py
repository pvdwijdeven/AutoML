# Standard library imports
import os
import pickle
from typing import Optional

# Third-party imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Local application imports
from automl.library import Logger
from automl.preprocessing import AutomlTransformer, preprocess

from .hypertuning import param_grids, param_grids_detailed
from .models import models
from .modelselection import (
    get_best_model_name,
    run_kfold_evaluation,
    run_kfold_grid_search,
    stacking_ensembler,
)
from .report import create_report
from .scoring import (
    flexible_scorer,
    get_scoring,
    lrmse_scorer,
    select_top_models,
    sort_ascending,
    write_to_output,
)


class AutomlModeling:

    def __init__(
        self,
        target: str,
        scoring: str,
        X_original: pd.DataFrame,
        y_original: Optional[pd.Series] = None,
        df_test: Optional[pd.DataFrame] = None,
        title: str = "",
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
        self.meta_data = {}
        self.meta_data["X_original"] = X_original
        self.meta_data["y_original"] = y_original
        self.meta_data["target"] = target
        self.meta_data["output_file"] = output_file
        self.meta_data["df_test"] = df_test
        self.meta_data["scoring"] = scoring
        self.meta_data["title"] = title
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

    def start_modelling(self) -> None:
        self.split_validation_data()

        (
            self.meta_data["X_val_prepro"],
            self.meta_data["y_val_prepro"],
            meta_data_add,
        ) = preprocess(
            X=self.meta_data["X_val"],
            y=self.meta_data["y_val"],
            logger=self.logger,
        )
        self.meta_data.update(meta_data_add)
        self.meta_data["X_train_val_size"] = self.meta_data["X_val"].shape
        self.meta_data["X_prepro_size"] = self.meta_data["X_val_prepro"].shape
        self.meta_data["scoring"] = get_scoring(
            scoring=self.meta_data["scoring"],
            dataset_type=self.meta_data["dataset_type"],
        )
        if self.meta_data["scoring"] == "lrmse":
            self.meta_data["scorer"] = lrmse_scorer
        else:
            self.meta_data["scorer"] = self.meta_data["scoring"]
        self.meta_data["scoring"] = self.meta_data["scoring"]
        self.meta_data["scorer"] = self.meta_data["scorer"]
        self.logger.info(
            msg=f"[BLUE] Dataset type: {self.meta_data["dataset_type"]}"
        )

        self.logger.info(msg="[MAGENTA] Starting primarily model selction")

        # Step 1 - model selection 10 models, 1 parameter per set
        checkpoint = self.load_step(step=1)
        if checkpoint is None:
            self.meta_data["10_models_output"] = run_kfold_evaluation(
                X=self.meta_data["X_val_prepro"],
                y=self.meta_data["y_val_prepro"],
                models=models,
                dataset_type=self.meta_data["dataset_type"],
                logger=self.logger,
                scoring=self.meta_data["scorer"],
            )
            self.meta_data["10_models_results"] = pd.DataFrame(
                [
                    {
                        "model": model,
                        "mean_score": details["mean_score"],
                        "std_score": details["std_score"],
                        "time_taken": details["time_taken"],
                    }
                    for model, details in self.meta_data[
                        "10_models_output"
                    ].items()
                ]
            )
            # Sort models by mean descending or ascending
            self.meta_data["10_models_results"] = self.meta_data[
                "10_models_results"
            ].sort_values(
                by="mean_score",
                ascending=sort_ascending(scorer_name=self.meta_data["scoring"]),
            )
            self.meta_data["top_selection_step1"] = select_top_models(
                summary_df=self.meta_data["10_models_results"],
                scorer=self.meta_data["scoring"],
            )
            self.save_step(step=1)
        else:
            self.meta_data = checkpoint

        self.logger.info(
            msg=f"[MAGENTA] Starting hypertuning top {len(self.meta_data["top_selection_step1"])} model selction"
        )
        # Step 2 model selection top (max) 5 of step 1, small hypertuning grid set

        checkpoint = self.load_step(2)
        if checkpoint is None:
            self.meta_data["topX_grid_results"] = run_kfold_grid_search(
                dataset_type=self.meta_data["dataset_type"],
                scoring=self.meta_data["scorer"],
                top_models=self.meta_data["top_selection_step1"],
                param_grid_matrix=param_grids,
                X=self.meta_data["X_val_prepro"],
                y=self.meta_data["y_val_prepro"],
                logger=self.logger,
            )
            self.save_step(step=2)
        else:
            self.meta_data = checkpoint

        (
            self.meta_data["TopX_best_model_name"],
            self.meta_data["TopX_best_score"],
        ) = get_best_model_name(results=self.meta_data["topX_grid_results"])
        self.logger.info(
            msg=f"[MAGENTA]Best model: {self.meta_data["TopX_best_model_name"]}, score: {self.meta_data["TopX_best_score"]}"
        )

        # Step 3 hyperparameter tuning on best model of step 2
        checkpoint = self.load_step(step=3)
        if checkpoint is None:
            self.meta_data["Top_model_top_grid_output"] = run_kfold_grid_search(
                dataset_type=self.meta_data["dataset_type"],
                scoring=self.meta_data["scorer"],
                top_models=[
                    {"model_name": self.meta_data["TopX_best_model_name"]}
                ],
                param_grid_matrix=param_grids_detailed,
                X=self.meta_data["X_val_prepro"],
                y=self.meta_data["y_val_prepro"],
                logger=self.logger,
            )
            self.save_step(step=3)
        else:
            self.meta_data = checkpoint

        # self.meta_data["step1"] = self.meta_data["10_models_output"]
        self.meta_data["top_selection"] = self.meta_data["top_selection_step1"]
        (
            self.meta_data["Top_model_best_model_name"],
            self.meta_data["Top_model_best_score"],
        ) = get_best_model_name(
            results=self.meta_data["Top_model_top_grid_output"]
        )
        self.logger.info(
            msg=f"Best model: {self.meta_data["Top_model_best_model_name"]}, score: {self.meta_data["Top_model_best_score"]}"
        )

        checkpoint = self.load_step(step=4)
        if checkpoint is None:
            self.meta_data["stacking_model"] = stacking_ensembler(
                meta_data=self.meta_data,
                X=self.meta_data["X_val_prepro"],
                y=self.meta_data["y_val_prepro"],
                logger=self.logger,
            )
            self.save_step(step=4)
        else:
            self.meta_data = checkpoint

        self.meta_data["Step5"] = self.meta_data["stacking_model"]
        # Step 4 -

        if "encode_target" in self.meta_data:
            self.meta_data["y_test_prepro"] = self.meta_data["encode_target"][
                "encoder"
            ].transform(y=self.meta_data["y_final_test"])
        elif "standardize_target" in self.meta_data:
            self.meta_data["y_test_prepro"] = self.meta_data[
                "standardize_target"
            ]["target_transformer"].transform(y=self.meta_data["y_final_test"])
        checkpoint = self.load_step(step=5)
        if checkpoint is None:
            self.meta_data["best_model"] = next(
                iter(self.meta_data["Top_model_top_grid_output"].values())
            )["best_estimator"].named_steps["model"]
            self.meta_data["best_estimator"] = next(
                iter(self.meta_data["Top_model_top_grid_output"].values())
            )["best_estimator"]
            self.meta_data["num_features"] = self.meta_data[
                "best_model"
            ].n_features_in_
            self.meta_data["transformer"] = AutomlTransformer(
                logger=self.logger
            )
            self.meta_data["transformer"].fit(
                X_train=self.meta_data["X_val_prepro"],
                y_train=self.meta_data["y_val_prepro"],
            )
            self.meta_data["X_val_trans"] = self.meta_data[
                "transformer"
            ].transform(X=self.meta_data["X_val_prepro"])
            self.meta_data["X_test_trans"] = self.meta_data[
                "transformer"
            ].transform(X=self.meta_data["X_final_test"])
            self.meta_data["best_model"].fit(
                self.meta_data["X_val_trans"], self.meta_data["y_val_prepro"]
            )
            self.meta_data["flex_scorer"] = flexible_scorer(
                estimator=self.meta_data["best_model"],
                X=self.meta_data["X_test_trans"],
                y=self.meta_data["y_test_prepro"],
                scorer_param=self.meta_data["scorer"],
            )
            self.meta_data["transformer"] = self.meta_data[
                "transformer"
            ].meta_data
            self.save_step(step=5)
        else:
            self.meta_data = checkpoint
        self.meta_data["num_features"] = self.meta_data[
            "best_model"
        ].n_features_in_
        self.logger.info(
            msg=f"[ORANGE] Scoring on 20% untouched dataset with {self.meta_data["TopX_best_model_name"]}: {self.meta_data["flex_scorer"]}"
        )
        self.meta_data["final_score"] = self.meta_data["flex_scorer"]

        # step 6: test on meta stacker
        self.meta_data["transformer"] = AutomlTransformer(logger=self.logger)
        self.meta_data["transformer"].fit(
            X_train=self.meta_data["X_val_prepro"],
            y_train=self.meta_data["y_val_prepro"],
        )
        self.meta_data["X_val_trans"] = self.meta_data["transformer"].transform(
            X=self.meta_data["X_val_prepro"]
        )
        self.meta_data["X_test_trans"] = self.meta_data[
            "transformer"
        ].transform(X=self.meta_data["X_final_test"])
        self.meta_data["best_model_stack"] = next(
            iter(self.meta_data["stacking_model"].values())
        )["best_estimator"].named_steps["stacking"]
        self.meta_data["best_estimator_stack"] = next(
            iter(self.meta_data["stacking_model"].values())
        )["best_estimator"]
        self.meta_data["best_model_stack"].fit(
            self.meta_data["X_val_trans"], self.meta_data["y_val_prepro"]
        )
        self.meta_data["flex_scorer"] = flexible_scorer(
            estimator=self.meta_data["best_model_stack"],
            X=self.meta_data["X_test_trans"],
            y=self.meta_data["y_test_prepro"],
            scorer_param=self.meta_data["scorer"],
        )
        self.logger.info(
            msg=f"[ORANGE] Scoring on 20% untouched dataset with meta stacker: {self.meta_data["flex_scorer"]}"
        )
        # final step if df_test is available
        if self.meta_data["df_test"] is not None:
            (
                self.meta_data["X_full_prepro"],
                self.meta_data["y_full_prepro"],
                self.meta_data["preprocess_step4"],
            ) = preprocess(
                X=self.meta_data["X_original"],
                y=self.meta_data["y_original"],
                logger=self.logger,
            )
            self.meta_data["best_estimator"].fit(
                self.meta_data["X_full_prepro"], self.meta_data["y_full_prepro"]
            )
            self.meta_data["y_pred"] = self.meta_data["best_estimator"].predict(
                self.meta_data["df_test"]
            )
            if "encode_target" in self.meta_data:
                self.meta_data["y_pred"] = self.meta_data["encode_target"][
                    "encoder"
                ].inverse_transform(y=self.meta_data["y_pred"])
            elif "standardize_target" in self.meta_data:
                self.meta_data["y_pred"] = self.meta_data["standardize_target"][
                    "target_transformer"
                ].inverse_transform(y=self.meta_data["y_pred"])
            self.meta_data["first_col"] = self.meta_data["df_test"].iloc[:, 0]
            # Create a DataFrame combining the first column and y_pred
            self.meta_data["submission"] = pd.DataFrame(
                data={
                    self.meta_data["first_col"].name: self.meta_data[
                        "first_col"
                    ],
                    self.meta_data["target"]: self.meta_data["y_pred"],
                }
            )

            # Save DataFrame to CSV
            self.meta_data["submission"].to_csv(
                self.meta_data["output_file"].replace(
                    "results.html", "submission.csv"
                ),
                index=False,
            )
            self.logger.info(
                f"Submissing saved to {self.meta_data["output_file"].replace("results.html", "submission.csv")}"
            )
            # same for meta stacker
            self.meta_data["best_estimator_stack"].fit(
                self.meta_data["X_full_prepro"], self.meta_data["y_full_prepro"]
            )
            self.meta_data["y_pred"] = self.meta_data[
                "best_estimator_stack"
            ].predict(self.meta_data["df_test"])
            if "encode_target" in self.meta_data:
                self.meta_data["y_pred"] = self.meta_data["encode_target"][
                    "encoder"
                ].inverse_transform(y=self.meta_data["y_pred"])
            elif "standardize_target" in self.meta_data:
                self.meta_data["y_pred"] = self.meta_data["standardize_target"][
                    "target_transformer"
                ].inverse_transform(y=self.meta_data["y_pred"])
            self.meta_data["first_col"] = self.meta_data["df_test"].iloc[:, 0]
            # Create a DataFrame combining the first column and y_pred
            self.meta_data["submission"] = pd.DataFrame(
                data={
                    self.meta_data["first_col"].name: self.meta_data[
                        "first_col"
                    ],
                    self.meta_data["target"]: self.meta_data["y_pred"],
                }
            )

            # Save DataFrame to CSV
            self.meta_data["submission"].to_csv(
                self.meta_data["output_file"].replace(
                    "results.html", "submission_stack.csv"
                ),
                index=False,
            )
            self.logger.info(
                f"Submissing for stack saved to {self.meta_data["output_file"].replace("results.html", "submission_stack.csv")}"
            )

        # Save meta_data as a text file
        meta_data_path = self.meta_data["output_file"].replace(
            "results.html", "meta_data.txt"
        )

        # Read the file in your custom format
        with open(file=meta_data_path, mode="w") as f:
            for key, value in self.meta_data.items():
                f.write(f"{key}: {value}\n")

        self.logger.info(msg=f"Meta data saved to {meta_data_path}")
        result = create_report(meta_data=self.meta_data)
        write_to_output(
            html=result,
            output_file=self.meta_data["output_file"],
        )
        self.logger.info(msg="[MAGENTA] DONE")

    def split_validation_data(
        self,
        random_state=42,
        test_size=0.2,
        shuffle=True,
    ) -> None:
        (
            self.meta_data["X_val"],
            self.meta_data["X_final_test"],
            self.meta_data["y_val"],
            self.meta_data["y_final_test"],
        ) = train_test_split(
            self.meta_data["X_original"],
            self.meta_data["y_original"],
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

    def save_step(self, step: int):

        path = self.meta_data["output_file"].replace(
            "results.html", f"step{step}.pkl"
        )
        with open(file=path, mode="wb") as f:
            pickle.dump(obj=self.meta_data, file=f)
        self.logger.info(msg=f"[GREEN] Saved checkpoint: {path}")

    def load_step(self, step: int):
        path = self.meta_data["output_file"].replace(
            "results.html", f"step{step}.pkl"
        )
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.logger.info(f"[CYAN] Loaded checkpoint: {path}")
                return pickle.load(f)
        return None
