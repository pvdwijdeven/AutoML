# Standard library imports
import time
from typing import Any, Callable

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline

# Local application imports
from automl.preprocessing import AutomlTransformer

from .models import model_class_map
from .scoring import sort_ascending


def run_kfold_evaluation(
    X, y, models, dataset_type, logger, scoring: str | Callable = "", folds=5
) -> dict[str, Any]:
    """
    Runs K-Fold cross-validation using AutomlTransformer and the models
    defined for the given dataset type.

    Parameters
    ----------
    X : numpy array or pandas DataFrame
        Feature matrix
    y : numpy array or pandas Series
        Target vector
    models : dict
        dictionary of models keyed by dataset_type
    dataset_type : str
        Dataset type key ("binary_classification", "imbalanced_binary_classification", "regression")
    folds : int
        Number of cross-validation folds

    Returns
    -------
    results : dict
        dictionary with model names as keys and CV score statistics as values
    """

    dataset_models = models.get(dataset_type, {})
    if not dataset_models:
        raise ValueError(f"No models found for dataset type: {dataset_type}")

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    results = {}

    for model_name, model in dataset_models.items():
        if model is None:
            continue  # skip if not implemented
        logger.info(f"[GREEN]- Training/scoring {model}")
        pipeline = Pipeline(
            [
                ("preprocessing", AutomlTransformer(logger=None)),
                ("model", model),
            ]
        )

        # Pick scoring metric
        start_time = time.perf_counter()
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring)
        end_time = time.perf_counter()
        results[model_name] = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "time_taken": end_time - start_time,
            "all_scores": scores,
        }

    return results


def run_kfold_grid_search(
    dataset_type,
    scoring,
    top_models,
    param_grid_matrix,
    X,
    y,
    logger,
    n_splits=5,
    random_state=42,
):
    """
    Run k-fold GridSearchCV for each model in top_models based on dataset_type.

    Parameters:
    - dataset_type (str): e.g. "binary_classification"
    - top_models (list of dict): each dict contains 'model_name' key
    - param_grid_matrix (dict): param grids for all dataset types and models
    - X, y: input features and labels
    - n_splits (int): number of cv folds, default 5
    - random_state (int): random state for reproducibility

    Returns:
    - results (dict): keys=model_name, values= best estimator and cv results
    """
    results = {}
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for model_info in top_models:
        model_name = model_info["model_name"]
        # Handle name differences in keys (case insensitive)
        model_name_key = next(
            (
                k
                for k in param_grid_matrix[dataset_type].keys()
                if k.lower() == model_name.lower()
            ),
            None,
        )
        if model_name_key is None:
            print(
                f"Warning: No param grid found for model '{model_name}' in dataset type '{dataset_type}'. Skipping."
            )
            continue

        model = model_class_map[model_name_key]

        pipeline = Pipeline(
            [
                ("preprocessing", AutomlTransformer(logger=None)),
                ("model", model),
            ]
        )

        param_grid = {}
        # Prefix 'model__' to each param name for Pipeline compatibility
        for param, values in param_grid_matrix[dataset_type][
            model_name_key
        ].items():
            param_grid[f"model__{param}"] = values

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        logger.info(f"[GREEN]- Running GridSearchCV for {model_name_key}...")
        start_time = time.perf_counter()
        grid_search.fit(X, y)
        end_time = time.perf_counter()
        results[model_name_key] = {
            "best_estimator": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
            "time_taken": end_time - start_time,
        }
        logger.info(
            f"[BLUE]Completed GridSearchCV for {model_name_key}. Best accuracy: {grid_search.best_score_:.4f}"
        )

    return results


def create_results_html_table(results):
    """
    Create an HTML table summarizing grid search results.

    Parameters:
    - results (dict): dictionary with keys as model names and values containing
      'best_params', 'best_score', 'cv_results' from GridSearchCV.

    Returns:
    - HTML string for a summary table.
    """
    summary_rows = []
    for model_name, res in results.items():
        best_params = res.get("best_params", {})
        best_score = res.get("best_score", None)
        mean_test_score = res.get("cv_results", {}).get("mean_test_score", [])
        std_test_score = res.get("cv_results", {}).get("std_test_score", [])

        # Optional: Extract number of candidates and best index info
        n_candidates = len(res.get("cv_results", {}).get("params", []))

        # Compose a readable summary string for best params
        best_params_str = ", ".join(
            f"{k.replace('model__','')}={v}" for k, v in best_params.items()
        )

        summary_rows.append(
            {
                "Model": model_name,
                "Best Score (Accuracy)": (
                    f"{best_score:.4f}" if best_score is not None else "N/A"
                ),
                "Best Parameters": best_params_str,
                "Mean test score": mean_test_score,
                "Standard test score": std_test_score,
                "Number of Candidates": n_candidates,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    # Make the HTML table with some style
    html_table = df_summary.to_html(
        index=False, classes="table table-striped", escape=False
    )
    return html_table


def get_best_model_name(results):
    """
    Get the model name with the highest best_score from GridSearch results.

    Parameters:
    - results (dict): keys are model names, values contain "best_score" keys.

    Returns:
    - best_model_name (str): model name with highest score
    - best_score (float): best score value
    """
    best_model_name = None
    best_score = float("-inf")

    for model_name, res in results.items():
        score = res.get("best_score", float("-inf"))
        if score > best_score:
            best_score = score
            best_model_name = model_name

    return best_model_name, best_score


def stacking_ensembler(
    meta_data,
    X,
    y,
    logger,
) -> dict[str, Any]:

    # Define base models (already hypertuned instances)
    # replace first model with step3 optimized one
    my_models = []
    for model in meta_data["step2"]:
        model_name = model
        model_params = meta_data["step2"][model]["best_params"]
        model_score = meta_data["step2"][model]["best_score"]
        my_models.append([model_name, model_params, model_score])
    my_models_sorted = sorted(
        my_models,
        key=lambda x: x[2],
        reverse=not sort_ascending(scorer_name=meta_data["scoring"]),
    )
    best_model = next(iter(meta_data["step3"]))
    model_params = meta_data["step3"][best_model]["best_params"]
    model_score = meta_data["step3"][best_model]["best_score"]
    my_models_sorted[0] = [best_model, model_params, model_score]

    base_models = []
    for cur_model in my_models_sorted:
        cleaned_params = {
            key.replace("model__", ""): value
            for key, value in cur_model[1].items()
        }
        model = model_class_map[cur_model[0]]
        model.set_params(**cleaned_params)
        base_models.append((cur_model[0], model))

    # Stacking ensemble
    if meta_data["supervised_learning_problem_type"] == "classification":
        meta_learner = LogisticRegression()
        stacking = StackingClassifier(
            estimators=base_models, final_estimator=meta_learner, cv=5
        )
        param_grid = {
            "stacking__final_estimator__penalty": ["l1", "l2"],  # 2 options
            "stacking__final_estimator__C": np.logspace(
                start=-2, stop=2, num=5
            ),  # 5 values -> total: 2*5=10
            "stacking__final_estimator__solver": [
                "lbfgs",
                "saga",
            ],  # 2 options -> total: 10*2=20
            "stacking__final_estimator__max_iter": [
                500,
                1000,
            ],  # 2 options -> total: 20*2=40
        }
    else:
        param_grid = {
            "stacking__final_estimator__alpha": [0.01, 0.1, 1.0, 10.0],  # 4
            "stacking__final_estimator__fit_intercept": [True, False],  # 2
            "stacking__final_estimator__solver": ["auto", "svd"],  # 2
        }
        meta_learner = Ridge()
        stacking = StackingRegressor(
            estimators=base_models, final_estimator=meta_learner, cv=5
        )

    # Full pipeline: preprocessing + stacking ensemble
    model_pipeline = Pipeline(
        steps=[
            ("preprocessing", AutomlTransformer(logger=None)),
            ("stacking", stacking),
        ]
    )
    results = {}

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=meta_data["scorer"],
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    logger.info("[GREEN]- Running GridSearchCV for stack...")
    start_time = time.perf_counter()
    grid_search.fit(X, y)
    end_time = time.perf_counter()
    results["meta"] = {
        "best_estimator": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
        "time_taken": end_time - start_time,
    }
    logger.info(
        f"[BLUE]Completed GridSearchCV for {"meta"}. Best accuracy: {grid_search.best_score_:.4f}"
    )

    return results
