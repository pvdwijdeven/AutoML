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
from typing import Tuple, Any, Optional

scoring_per_dataset_type = {
    "binary_classification": "accuracy",
    "regression": "lrmse",
}

scoring_requirement = {
    "accuracy": "high",
    "lrmse": "low",
}


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


def summarize_results(
    results_dict, model_dict, scoring="accuracy"
) -> Tuple[
    pd.DataFrame, Optional[Any], Optional[str], Optional[float]
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

    # Flatten multi-index columns
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Find the best model based on mean "scoring"
    if f"{scoring}_mean" in summary.columns:
        if scoring_requirement[scoring] == "high":
            best_row = summary.loc[summary[f"{scoring}_mean"].idxmax()]
        else:
            best_row = summary.loc[summary[f"{scoring}_mean"].idxmin()]
        best_model_name = best_row["modelname"]
        assert isinstance(best_model_name, str)
        best_model = model_dict[best_row["modelname"]]
        best_accuracy = best_row[f"{scoring}_mean"]
        assert isinstance(best_accuracy, float)
    else:
        best_model, best_model_name, best_accuracy = None, None, None

    return summary, best_model, best_model_name, best_accuracy


def write_to_output(output_file, summary_df) -> None:

    html_table: str = summary_df.to_html(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    with open(output_file, "w") as f:
        f.write(html_table)
