from typing import Dict, Any
import pandas as pd
import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


def create_report(meta_data: Dict[str, Any]) -> str:
    overview = create_general_table(meta_data=meta_data)
    result = render_report(title=meta_data["title"], overview=overview)
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
    result = ""
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
    step1 = ""
    step2 = ""
    step3 = ""
    step4 = ""
    html = "<h3>General info:</h3>"
    html += general
    html += "<h3>Step1: model selection</h3>"
    html += step1
    html += "<h3>Step2: hypertuning top X</h3>"
    html += step2
    html += "<h3>Step3: Detailed hypertuning</h3>"
    html += step3
    html += "<h3>Step4: Ontouched test set</h3>"
    html += step4
    return html
