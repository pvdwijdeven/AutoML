import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
from math import sqrt
import numpy as np
from typing import Tuple, Any, List, Dict, Callable

from sklearn.metrics import make_scorer, get_scorer


def sort_ascending(scorer_name):
    """
    Determine if higher scorer values indicate better performance.

    Args:
        scorer_name (str): The name of the scorer/metric.

    Returns:
        bool: True if higher score is better, False otherwise.
    """
    # Lowercase for case-insensitive matching
    name = scorer_name.lower()

    # Common "higher is better" metrics
    high_better = [
        "neg_mean_squared_error",
        "neg_log_loss",
        "neg_mean_absolute_error",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "r2",
        "explained_variance",
        "balanced_accuracy",
    ]

    # Common "lower is better" metrics
    low_better = [
        # sklearn's negated MSE, already negative
        "mean_squared_error",
        "mean_absolute_error",
        "root_mean_squared_error",
        "log_loss",
        "lrsme",
    ]

    # Check contains any high_better substring
    for metric in high_better:
        if metric in name:
            return False

    # Check contains any low_better substring
    for metric in low_better:
        if metric in name:
            return True

    # Default fallback: assume higher is better
    return False


scoring_per_dataset_type = {
    "binary_classification": "accuracy",
    "regression": "lrmse",
}

scoring_requirement = {
    "accuracy": "high",
    "lrmse": "low",
}


def flexible_scorer(estimator, X, y, scorer_param):
    """
    Wrapper function to score estimator on (X, y) using scorer_param.
    scorer_param can be:
      - a string (built-in scorer name)
      - a callable scorer object with signature scorer(estimator, X, y)
    """
    if isinstance(scorer_param, str):
        # Use scikit-learn's get_scorer to retrieve scorer by name
        scorer_func = get_scorer(scorer_param)
        return scorer_func(estimator, X, y)
    elif callable(scorer_param):
        # If it's a custom callable scorer, call it directly
        return scorer_param(estimator, X, y)
    else:
        raise ValueError("scorer_param must be a string or a callable scorer")


# Define LRMSE scorer
def lrmse(y_true, y_pred):
    return np.sqrt(np.mean((np.log(y_true) - np.log(y_pred)) ** 2))


lrmse_scorer: Callable = make_scorer(score_func=lrmse, greater_is_better=False)


def get_score(
    dataset_type: str,
    X_test: pd.DataFrame,
    y_test,
    y_pred,
    model,
    name,
    end_time,
    start_time,
):
    results = {}
    # Choose metric depending on dataset type
    if dataset_type in (
        "binary_classification",
        "imbalanced_binary_classification",
    ):

        # You can choose metrics as needed; here we use accuracy and F1 + AUC if possible
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        f1 = f1_score(y_true=y_test, y_pred=y_pred)

        # Try to get predicted probabilities for AUC if supported
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_true=y_test, y_score=y_proba)
        except ValueError:
            auc = None

        results[name] = {
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
            "time": end_time - start_time,
        }

    elif dataset_type == "multi_class_classification":
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average="weighted")
        results[name] = {
            "accuracy": acc,
            "f1_score_weighted": f1,
            "time": end_time - start_time,
        }

    elif dataset_type == "multi_label_classification":
        # For multi-label, use an appropriate metric such as average F1 per label
        f1_macro = f1_score(y_true=y_test, y_pred=y_pred, average="macro")
        f1_micro = f1_score(y_true=y_test, y_pred=y_pred, average="micro")
        results[name] = {
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "time": end_time - start_time,
        }

    elif dataset_type == "ordinal_regression":
        # Could treat as regression or classification; here treat as classification
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        results[name] = {
            "accuracy": acc,
            "time": end_time - start_time,
        }

    elif dataset_type == "regression":
        epsilon = 1e-15  # small number to avoid log(0), change if needed
        lrmse: float = sqrt(
            mean_squared_error(
                y_true=np.log(y_test + epsilon),
                y_pred=np.log(np.maximum(y_pred, epsilon)),
            ),
        )

        mse: float = mean_squared_error(y_true=y_test, y_pred=y_pred)
        rmse: float = sqrt(mse)
        r2: float = r2_score(y_true=y_test, y_pred=y_pred)
        results[name] = {
            "lrmse": lrmse,
            "rmse": rmse,
            "mse": mse,
            "r2": r2,
            "time": end_time - start_time,
        }

    else:
        # Fallback metric if unknown
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
        results[name] = {
            "mse": mse,
            "time": end_time - start_time,
        }

    return results


def select_top_models(
    summary_df, scorer="accuracy", k=1, max_models=5, min_models=3
) -> List[Dict[str, Any]]:
    """
    Select top models based on mean and std accuracy criteria.

    Parameters:
    - summary_df: DataFrame with multi-index columns [(metric, agg), ...] grouped by modelname
    - models_dict: original dict with modelname as keys and model objects as values
    - metric: string, which metric to consider for ordering (e.g., 'accuracy')
    - k: int or float, how many std deviations below best mean to allow
    - max_models: int, max number of models to select

    Returns:
    - dict of selected models:
        {modelname: {'model': model_obj, 'accuracy_mean': float, 'accuracy_std': float}, ...}
    """

    # Extract mean and std columns for the metric, flatten columns for ease

    mu_best = summary_df.iloc[0]["mean_score"]
    sigma_best = summary_df.iloc[0]["std_score"]

    selected = []
    models = []
    for _, row in summary_df.iterrows():
        if row["mean_score"] >= mu_best - k * sigma_best:
            selected_model = {
                "model_name": row["model"],
                "accuracy_mean": row["mean_score"],
                "accuracy_std": row["std_score"],
            }
            models.append(row["model"])
            selected.append(selected_model)
            if len(selected) >= max_models:
                break

    if len(selected) < 3:
        for _, row in summary_df.iterrows():

            if row["model"] not in models:
                selected_model = {
                    "model_name": row["model"],
                    "accuracy_mean": row["mean_score"],
                    "accuracy_std": row["std_score"],
                }
                selected.append(selected_model)
                if len(selected) >= min_models:
                    break

    return selected


def summarize_results(
    results_dict, model_dict, scoring="accuracy"
) -> Tuple[
    pd.DataFrame, List[Dict]
]:  # -> tuple[DataFrame, Any | None, Any | Series[Any] | None, An...:
    # Flatten nested dictionary into a list of rows
    rows = []
    for fold, models in results_dict.items():
        for model, scores in models.items():
            row = {"fold": fold, "modelname": model}
            row.update(scores)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate mean and std per model and score
    summary = df.groupby("modelname").agg(
        {
            col: ["mean", "std"]
            for col in df.columns
            if col not in ["fold", "model", "modelname"]
        }
    )

    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    mean_col = f"{scoring}_mean"
    # std_col = f"{scoring}_std"

    # Sort models by mean descending
    sorted_summary = summary.sort_values(by=mean_col, ascending=False)

    return sorted_summary, select_top_models(
        summary_df=sorted_summary, k=1, max_models=5
    )


def get_scoring(scoring: str = "", dataset_type: str = "") -> str:
    if scoring == "":
        if dataset_type in [
            "binary_classification",
            "imbalanced_binary_classification",
        ]:
            scoring = "roc_auc"
        elif dataset_type == "regression":
            scoring = "r2"
        else:
            scoring = "accuracy"
    return scoring


def write_to_output(html, output_file) -> None:

    with open(output_file, "w") as f:
        f.write(html)
