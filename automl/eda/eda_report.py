from jinja2 import Environment, FileSystemLoader
import os
from automl.dataloader import ConfigData
from .dataset_overview import DatasetInfo
from datetime import datetime

def create_report(config_data: ConfigData, data_set_info: DatasetInfo) -> None:
    context = {"data": data_set_info.model_dump(), "header":"Dataset overview"}
    env = Environment(
        loader=FileSystemLoader(
            searchpath=os.path.join(config_data.root, "automl/jinja")
        )
    )
    template = env.get_template(name="table_from_dict.j2")
    html_dataset_info = template.render(**context)
    tabs = [
        {"title": "General overview", "content": html_dataset_info},
    ]
    template = env.get_template(name="report_main.j2")
    output_html = template.render(
            tabs=tabs,
            title=f"EDA Report {config_data.project_name}",
            current_time=datetime.now(),
        )
    with open(
        file=config_data.report_template, mode="w", encoding="utf-8"
    ) as f:
        f.write(output_html)
