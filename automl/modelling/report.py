from .models import models
from .scoring import sort_ascending
from .hypertuning import param_grids, param_grids_detailed
from typing import Dict, Any
import pandas as pd
import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import html
import inspect


def create_report(meta_data: Dict[str, Any]) -> str:
    general = create_general_table(meta_data=meta_data)
    order_list = [m["model_name"] for m in meta_data["top_selection"]]
    step1 = step1_to_html(
        results_dict=meta_data["step1"],
        models_dict=models[meta_data["dataset_type"]],
        order_list=order_list,
    )
    step2 = step2_to_html(meta_data=meta_data)
    step3 = step3_to_html(meta_data=meta_data)
    step4 = f"Scoring on 20% untouched dataset:<span class='highlight-yellow'>{meta_data["final_score"]:.6f}</span>"
    html = "<br><h3>General info:</h3>"
    html += general
    html += "<br><h3>Step1: model selection</h3>"
    html += step1
    html += f"<br><h3>Step2: hypertuning top {len(meta_data['step2'])}</h3>"
    html += step2
    html += "<br><h3>Step3: Detailed hypertuning</h3>"
    html += step3
    html += "<br><h3>Step4: Ontouched test set</h3>"
    html += step4

    result = render_report(title=meta_data["title"], overview=html)
    return result


def render_report(title, overview, details=""):
    tabs = [
        {"title": "General overview", "content": overview},
        {"title": "Details", "content": details},
    ]
    # Load and render the template
    current_dir = os.path.dirname(os.path.abspath(__file__))  # automl/gui
    automl_dir = os.path.dirname(current_dir)  # automl
    templates_dir = os.path.join(automl_dir, "templates")  # automl/templates
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("model_report.j2")
    output_html = template.render(
        tabs=tabs,
        title=f"EDA Report {title}",
        current_time=datetime.now(),
    )
    return output_html


def create_general_table(meta_data: Dict[str, Any]) -> str:
    problem_type = f"{meta_data["supervised_learning_problem_type"]} ({meta_data["dataset_type"]})"
    general_dict = {
        "Problem type": problem_type,
        "Target": meta_data["target"],
        "Size before preprocessing": f"{meta_data["X_train_val_size"][1]} x {meta_data["X_train_val_size"][0]}",
        "Size after preprocessing": f"{meta_data["num_features"]} x {meta_data["X_prepro_size"][0]}",
        "Scoring method": meta_data["scoring"],
    }
    df = pd.DataFrame(data=general_dict, index=[0]).transpose()
    df = df.reset_index()  # Turn the index into a column
    df.columns = ["Description", "Value"]  # Optionally rename columns
    general = df.to_html(index=False, header=False)

    return general


def get_explicit_params(estimator):
    """
    Return only the parameters that differ from the default values
    (i.e., explicitly set in the constructor).
    """
    params = estimator.get_params(deep=False)
    explicit_params = {}

    # Inspect the class constructor to get defaults
    sig = inspect.signature(estimator.__class__.__init__)
    defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    # Keep only params that differ from defaults
    for k, val in params.items():
        if k in defaults and val != defaults[k]:
            explicit_params[k] = val

    return explicit_params


def step1_to_html(
    results_dict,
    models_dict=None,
    order_list=None,
    max_param_length=120,
):
    """
    Convert model evaluation results into an HTML table, sorted by a given order,
    including only explicitly set parameters and a 'Selected' column.

    Parameters:
        results_dict (dict): Model scores, times, etc.
        models_dict (dict, optional): Dict mapping model names to estimator objects.
        order_list (list, optional): List of model names defining the display order.
        max_param_length (int): Maximum characters for the parameter string.

    Returns:
        str: HTML table string.
    """

    headers = ["Model", "Mean Score", "Std Score"]
    if models_dict is not None:
        headers.append("Parameters")
    headers.append("Time Taken (sec)")
    html_table = '<table border="1" cellpadding="5" cellspacing="0">\n'
    html_table += (
        "  <tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>\n"
    )

    # Define order of models
    if order_list is not None:
        ordered_models = [m for m in order_list if m in results_dict]
        # Add remaining models not in order_list
        remaining = [m for m in results_dict.keys() if m not in order_list]
        ordered_models.extend(remaining)
    else:
        ordered_models = list(results_dict.keys())

    for model in ordered_models:
        metrics = results_dict[model]
        mean_score = float(metrics["mean_score"])
        std_score = float(metrics["std_score"])
        time_taken = float(metrics["time_taken"]) / 5

        param_str = ""
        if models_dict and model in models_dict:
            estimator = models_dict[model]
            if hasattr(estimator, "get_params"):
                params = get_explicit_params(estimator)
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                if len(param_str) > max_param_length:
                    param_str = param_str[:max_param_length] + "..."
        highlight_start = (
            '<span class="highlight-yellow">'
            if order_list and model in order_list
            else ""
        )
        highlight_end = "</span>" if order_list and model in order_list else ""
        row_html = (
            f"  <tr>"
            f"<td>{highlight_start}{html.escape(model)}{highlight_end}</td>"
            f"<td>{mean_score:.6f}</td>"
            f"<td>{std_score:.6f}</td>"
        )

        if models_dict:
            row_html += f"<td>{html.escape(param_str)}</td>"
        row_html += f"<td>{time_taken:.3f} sec</td>"

        row_html += "</tr>\n"
        html_table += row_html

    html_table += "</table>"
    html_table += "<span class='highlight-yellow'><sup>model selected for next step</sup></span><br>"
    return html_table


def step2_to_html(
    meta_data,
):
    """
    Convert model evaluation results into an HTML table, sorted by a given order,
    including only explicitly set parameters and a 'Selected' column.

    Parameters:
        meta_data (dict(str,dict)): class meta_data


    Returns:
        str: HTML table string.
    """

    headers = [
        "Model",
        "Mean Score",
        "Best params",
        "Number of runs",
        "Grid params",
        "Time Taken (sec)",
    ]

    html_table = '<table border="1" cellpadding="5" cellspacing="0">\n'
    html_table += (
        "  <tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>\n"
    )

    rows = []
    first = True
    for model in meta_data["step2"]:

        best_score = meta_data["step2"][model]["best_score"]
        best_params_temp = meta_data["step2"][model]["best_params"]
        number_of_runs = len(
            meta_data["step2"][model]["cv_results"]["mean_fit_time"]
        )
        time_taken = meta_data["step2"][model]["time_taken"]
        best_params = []
        for param in best_params_temp:
            my_param = param.replace("model__", "")
            best_params.append(f"{my_param}:{best_params_temp[param]}")
        grid = param_grids[meta_data["dataset_type"]][model]
        model_grid = []
        for param in grid:
            model_grid.append(f"{param}:{grid[param]}")

        rows.append(
            {
                "model": model,
                "best_score": best_score,
                "best_params": best_params,
                "number_of_runs": number_of_runs,
                "model_grid": model_grid,
                "time_taken": time_taken,
            }
        )

    # Sort rows by best_score descending (highest score first)
    rows = sorted(
        rows,
        key=lambda x: x["best_score"],
        reverse=sort_ascending(scorer_name=meta_data["scoring"]),
    )

    html_output = ""
    for row in rows:
        highlight_start = '<span class="highlight-yellow">' if first else ""
        highlight_end = "</span>" if first else ""
        first = False
        html_output += (
            f"  <tr>"
            f"<td>{highlight_start}{html.escape(row['model'])}{highlight_end}</td>"
            f"<td>{row['best_score']:.6f}</td>"
            f"<td>{'<br>'.join(row['best_params'])}</td>"
            f"<td>{row['number_of_runs']}</td>"
            f"<td>{'<br>'.join(row['model_grid'])}</td>"
            f"<td>{row['time_taken']:.3f} sec</td>"
            f"</tr>\n"
        )

        html_output += "</tr>\n"
    html_table += html_output

    html_table += "</table>"
    html_table += "<span class='highlight-yellow'><sup>model selected for next step</sup></span><br>"
    return html_table


def step3_to_html(
    meta_data,
):
    """
    Convert model evaluation results into an HTML table, sorted by a given order,
    including only explicitly set parameters and a 'Selected' column.

    Parameters:
        meta_data (dict(str,dict)): class meta_data


    Returns:
        str: HTML table string.
    """

    headers = [
        "Model",
        "Mean Score",
        "Best params",
        "Number of runs",
        "Grid params",
        "Time Taken (sec)",
    ]

    html_table = '<table border="1" cellpadding="5" cellspacing="0">\n'
    html_table += (
        "  <tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>\n"
    )

    rows = []
    for model in meta_data["step3"]:

        best_score = meta_data["step3"][model]["best_score"]
        best_params_temp = meta_data["step3"][model]["best_params"]
        number_of_runs = len(
            meta_data["step3"][model]["cv_results"]["mean_fit_time"]
        )
        time_taken = meta_data["step3"][model]["time_taken"]
        best_params = []
        for param in best_params_temp:
            my_param = param.replace("model__", "")
            best_params.append(f"{my_param}:{best_params_temp[param]}")
        grid = param_grids_detailed[meta_data["dataset_type"]][model]
        model_grid = []
        for param in grid:
            model_grid.append(f"{param}:{grid[param]}")

        rows.append(
            {
                "model": model,
                "best_score": best_score,
                "best_params": best_params,
                "number_of_runs": number_of_runs,
                "model_grid": model_grid,
                "time_taken": time_taken,
            }
        )

    # Sort rows by best_score descending (highest score first)
    rows = sorted(rows, key=lambda x: x["best_score"], reverse=True)
    first = True
    html_output = ""
    for row in rows:
        highlight_start = '<span class="highlight-yellow">' if first else ""
        highlight_end = "</span>" if first else ""
        first = False
        html_output += (
            f"  <tr>"
            f"<td>{highlight_start}{html.escape(row['model'])}{highlight_end}</td>"
            f"<td>{row['best_score']:.6f}</td>"
            f"<td>{'<br>'.join(row['best_params'])}</td>"
            f"<td>{row['number_of_runs']}</td>"
            f"<td>{'<br>'.join(row['model_grid'])}</td>"
            f"<td>{row['time_taken']:.3f} sec</td>"
            f"</tr>\n"
        )

        html_output += "</tr>\n"
    html_table += html_output

    html_table += "</table>"
    return html_table
