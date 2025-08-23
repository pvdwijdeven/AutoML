# internal libraries
from preprocessing import preprocess, AutomlTransformer
from library import Logger
from .models import models
from .modelselection import (
    run_kfold_evaluation,
    run_kfold_grid_search,
    get_best_model_name,
    stacking_ensembler,
)
from .scoring import (
    select_top_models,
    write_to_output,
    get_scoring,
    lrmse_scorer,
    flexible_scorer,
    sort_ascending,
)
from .report import create_report
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
        self.X_original = X_original
        self.y_original = y_original
        self.target: str = target
        self.output_file: str = output_file
        self.df_test = df_test
        self.scoring = scoring
        self.title = title
        self.meta_data = {}
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
        self.meta_data["X_train_val_size"] = self.X_val.shape
        self.meta_data["X_prepro_size"] = self.X_val_prepro.shape
        self.meta_data["target"] = self.target
        self.meta_data["title"] = self.title
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
                        "time_taken": details["time_taken"],
                    }
                    for model, details in results.items()
                ]
            )
            # Sort models by mean descending or ascending
            df_results = df_results.sort_values(
                by="mean_score",
                ascending=sort_ascending(scorer_name=self.scoring),
            )
            top_selection = select_top_models(
                summary_df=df_results, scorer=self.scoring
            )
            self.save_step(step=1, obj=(df_results, results, top_selection))
        else:
            df_results, results, top_selection = checkpoint

        self.logger.info(
            msg=f"[MAGENTA] Starting hypertuning top {len(top_selection)} model selction"
        )
        # Step 2 model selection top (max) 5 of step 1, small hypertuning grid set

        checkpoint = self.load_step(2)
        if checkpoint is None:
            best_grid_model = run_kfold_grid_search(
                dataset_type=self.dataset_type,
                scoring=self.scorer,
                top_models=top_selection,
                param_grid_matrix=param_grids,
                X=self.X_val_prepro,
                y=self.y_val_prepro,
                logger=self.logger,
            )
            self.save_step(step=2, obj=(best_grid_model))
        else:
            best_grid_model = checkpoint

        best_model_name, best_score = get_best_model_name(
            results=best_grid_model
        )
        self.logger.info(
            msg=f"[MAGENTA]Best model: {best_model_name}, score: {best_score}"
        )

        # Step 3 hyperparameter tuning on best model of step 2
        checkpoint = self.load_step(step=3)
        if checkpoint is None:
            fine_tuned_model = run_kfold_grid_search(
                dataset_type=self.dataset_type,
                scoring=self.scorer,
                top_models=[{"model_name": best_model_name}],
                param_grid_matrix=param_grids_detailed,
                X=self.X_val_prepro,
                y=self.y_val_prepro,
                logger=self.logger,
            )
            self.save_step(step=3, obj=(fine_tuned_model))
        else:
            fine_tuned_model = checkpoint

        self.meta_data["step1"] = results
        self.meta_data["top_selection"] = top_selection
        self.meta_data["step2"] = best_grid_model
        self.meta_data["step3"] = fine_tuned_model
        best_model_name, best_score = get_best_model_name(
            results=fine_tuned_model
        )
        self.logger.info(
            msg=f"Best model: {best_model_name}, score: {best_score}"
        )
        result = stacking_ensembler(
            meta_data=self.meta_data,
            X=self.X_val_prepro,
            y=self.y_val_prepro,
            logger=self.logger,
        )
        self.meta_data["Step5"] = result
        # Step 5 - final test on unexposed test set on best model of step 3/4.

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
            best_model = next(iter(fine_tuned_model.values()))[
                "best_estimator"
            ].named_steps["model"]
            best_estimator = next(iter(fine_tuned_model.values()))[
                "best_estimator"
            ]
            self.meta_data["num_features"] = best_model.n_features_in_
            transformer = AutomlTransformer(logger=self.logger)
            transformer.fit(
                X_train=self.X_val_prepro, y_train=self.y_val_prepro
            )
            self.X_val_trans = transformer.transform(X=self.X_val_prepro)
            self.X_test_trans = transformer.transform(X=self.X_final_test)
            best_model.fit(self.X_val_trans, self.y_val_prepro)
            score = flexible_scorer(
                estimator=best_model,
                X=self.X_test_trans,
                y=self.y_test_prepro,
                scorer_param=self.scorer,
            )
            self.meta_data["transformer"] = transformer.meta_data
            self.save_step(
                step=4,
                obj=(best_model, best_estimator, score, transformer.meta_data),
            )
        else:
            best_model, best_estimator, score, self.meta_data["transformer"] = (
                checkpoint
            )
        self.meta_data["num_features"] = best_model.n_features_in_
        self.logger.info(
            msg=f"[ORANGE] Scoring on 20% untouched dataset: {score}"
        )
        self.meta_data["final_score"] = score
        # final step if df_test is available
        if self.df_test is not None:
            self.X_full_prepro, self.y_full_prepro, self.meta_data_add = (
                preprocess(
                    X=self.X_original, y=self.y_original, logger=self.logger
                )
            )
            self.meta_data["preprocess_step4"] = self.meta_data_add
            best_estimator.fit(self.X_full_prepro, self.y_full_prepro)
            self.y_pred = best_estimator.predict(self.df_test)
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

        # Save meta_data as a text file
        meta_data_path = self.output_file.replace(
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
            output_file=self.output_file,
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
