from preprocessing import AutomlTransformer
from .models import model_class_map
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import numpy as np
import pandas as pd


def run_kfold_evaluation(
    X, y, models, dataset_type, logger, scoring="", folds=5
):
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
        Dictionary of models keyed by dataset_type
    dataset_type : str
        Dataset type key ("binary_classification", "imbalanced_binary_classification", "regression")
    folds : int
        Number of cross-validation folds

    Returns
    -------
    results : dict
        Dictionary with model names as keys and CV score statistics as values
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

        scores = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring)
        results[model_name] = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "all_scores": scores,
        }

    return results


def run_kfold_grid_search(
    dataset_type,
    scoring,
    top_models,
    param_grids,
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
    - param_grids (dict): param grids for all dataset types and models
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
                for k in param_grids[dataset_type].keys()
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
        for param, values in param_grids[dataset_type][model_name_key].items():
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
        grid_search.fit(X, y)

        results[model_name_key] = {
            "best_estimator": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
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
