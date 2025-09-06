# Standard library imports
import os
from dataclasses import asdict
from datetime import datetime

# Third-party imports
from jinja2 import Environment, FileSystemLoader

# Local application imports
from automl.dataloader import ConfigData

from .dataset_overview import DatasetInfo


def create_report(config_data: ConfigData, data_set_info: DatasetInfo) -> None:
    context = {"data": asdict(obj=data_set_info), "header":"Dataset overview"}
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
