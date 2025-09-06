# Standard library imports
import os
import re
from dataclasses import asdict
from datetime import datetime

# Third-party imports
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# Local application imports
from automl.dataloader import ConfigData, OriginalData

from .dataset_overview import DatasetInfo


def add_links_to_headers(html: str, target: str) -> str:
    def replace_th(match) -> str:
        col_name = match.group(1)
        if col_name == target:
            link = f'<a href="#target_{col_name.replace(" ", "-")}" onclick="showTab(2)" class="feature-link">{col_name}</a>'
        else:
            link = f'<a href="#feature_{col_name.replace(" ", "-")}" onclick="showTab(1)" class="feature-link">{col_name}</a>'
        return f"<th>{link}</th>"

    # Replace every <th>...</th> with a linked version
    return re.sub(r"<th>(.*?)</th>", replace_th, html).replace('border="1"', "")


def create_general_overview(
    env: Environment,
    config_data: ConfigData,
    data_set_info: DatasetInfo,
    original_data: OriginalData,
) -> str:
    # generate overview table
    dataset_overview = {
        "data": asdict(obj=data_set_info),
        "header": "Dataset overview",
    }
    template = env.get_template(name="table_from_dict.j2")
    html_dataset_overview = template.render(**dataset_overview)

    X_total = pd.concat(
        objs=[original_data.X_train, original_data.y_train], axis=1
    )
    # generate sample data
    samples_head = add_links_to_headers(
        html=X_total.head(n=10).to_html(
            index=False, na_rep="<N/A>", classes="table table-striped"
        ),
        target=str(object=config_data.target),
    )
    samples_middle = add_links_to_headers(
        html=X_total.iloc[
            data_set_info.number_of_samples // 2
            - 5 : data_set_info.number_of_samples // 2
            + 5
        ].to_html(index=False, na_rep="<N/A>", classes="table table-striped"),
        target=str(object=config_data.target),
    )
    samples_tail = add_links_to_headers(
        html=X_total.tail(n=10).to_html(
            index=False, na_rep="<N/A>", classes="table table-striped"
        ),
        target=str(object=config_data.target),
    )
    sample_data = {
        "samples_head": samples_head,
        "samples_middle": samples_middle,
        "samples_tail": samples_tail,
        "header": "Sample data train-set",
    }
    template = env.get_template(name="sample_data.j2")
    html_sample_data = template.render(**sample_data)

    return f"{html_dataset_overview}{html_sample_data}"


def create_report(
    config_data: ConfigData,
    data_set_info: DatasetInfo,
    original_data: OriginalData,
) -> None:
    # load jinja2 environment
    env = Environment(
        loader=FileSystemLoader(
            searchpath=os.path.join(config_data.root, "automl/jinja")
        )
    )

    html_general_overview = create_general_overview(
        env=env,
        config_data=config_data,
        data_set_info=data_set_info,
        original_data=original_data,
    )

    # complete report
    tabs = [
        {"title": "General overview", "content": html_general_overview},
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
