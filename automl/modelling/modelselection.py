from preprocessing import AutomlTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


def run_kfold_evaluation(X, y, models, dataset_type, logger, folds=5):
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
                ("preprocessing", AutomlTransformer(logger=logger)),
                ("model", model),
            ]
        )

        # Pick scoring metric
        if dataset_type in [
            "binary_classification",
            "imbalanced_binary_classification",
        ]:
            scoring = "roc_auc"
        elif dataset_type == "regression":
            scoring = "r2"
        else:
            scoring = "accuracy"

        scores = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring)
        results[model_name] = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "all_scores": scores,
        }

    return results
