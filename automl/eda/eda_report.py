# Standard library imports
import os
import re
from dataclasses import asdict
from datetime import datetime
from typing import Union

# Third-party imports
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# Local application imports
from automl.dataloader import ConfigData, OriginalData
from .dataset_overview import DatasetInfo
from .column_analysis import ColumnInfoMapping, ColumnPlotMapping
from .target_relations import TargetRelationMapping
from .relations import RelationInfoMapping
from .missing import MissingOverview
from .testdata import TestInfo


def add_links_to_headers(html: str) -> str:
    def replace_th(match) -> str:
        col_name = match.group(1)
        link = f'<a href="#feature_{col_name.replace(" ", "-")}" onclick="showTab(1)" class="feature-link">{col_name}</a>'
        return f"<th>{link}</th>"

    # Replace every <th>...</th> with a linked version
    return re.sub(pattern=r"<th>(.*?)</th>", repl=replace_th, string=html).replace('border="1"', "")


def create_feature_report(
    env: Environment, columninfo: ColumnInfoMapping, plotinfo: ColumnPlotMapping
) -> str:
    template = env.get_template(name="feature_tables.j2")
    return template.render(
        **{"columninfo_mapping": columninfo}, **{"columnplot_mapping": plotinfo}
    )


def create_test_report(
    env: Environment, test_info: TestInfo
) -> str:
    template = env.get_template(name="test_info.j2")
    return template.render(**{"test_info": test_info})


def create_missing_report(
    env: Environment, missing_info: MissingOverview
) -> str:
    template = env.get_template(name="missing_info.j2")
    return template.render(**{"missinginfo": missing_info})


def create_target_relations_report(
    env: Environment, target_relations: TargetRelationMapping
) -> str:
    template = env.get_template(name="target_relations.j2")
    return template.render(**{"columninfo_mapping": target_relations})


def create_relations_report(
    env: Environment, relation_info: RelationInfoMapping
) -> str:
    template = env.get_template(name="relations.j2")
    return template.render(**{"relation_info": relation_info})


def create_general_overview(
    env: Environment,
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
    )
    samples_middle = add_links_to_headers(
        html=X_total.iloc[
            data_set_info.number_of_samples // 2
            - 5 : data_set_info.number_of_samples // 2
            + 5
        ].to_html(index=False, na_rep="<N/A>", classes="table table-striped"),
    )
    samples_tail = add_links_to_headers(
        html=X_total.tail(n=10).to_html(
            index=False, na_rep="<N/A>", classes="table table-striped"
        ),
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
    column_info: ColumnInfoMapping,
    column_plot: ColumnPlotMapping,
    target_relations: TargetRelationMapping,
    relation_info: RelationInfoMapping,
    missing_info: MissingOverview,
    test_info: Union[TestInfo,None],
) -> None:
    # load jinja2 environment
    env = Environment(
        loader=FileSystemLoader(
            searchpath=os.path.join(config_data.root, "automl/jinja")
        )
    )

    html_general_overview = create_general_overview(
        env=env,
        data_set_info=data_set_info,
        original_data=original_data,
    )

    html_features = create_feature_report(
        env=env, columninfo=column_info, plotinfo=column_plot
    )

    html_target_relations = create_target_relations_report(
        env=env, target_relations=target_relations
    )

    html_relations = create_relations_report(
        env=env, relation_info=relation_info
    )

    html_missing = create_missing_report(env=env, missing_info=missing_info)
    if test_info is not None:
        html_test = create_test_report(env=env, test_info=test_info)
    else:
        html_test = ""
    # complete report
    tabs = [
        {
            "title": "General overview",
            "content": html_general_overview,
        }, 
        {"title": "Features", "content": html_features},
        {"title": "Target relations", "content": html_target_relations},
        {"title": "Relations", "content": html_relations},
        {"title": "Missing", "content": html_missing},
        {"title": "Test Data", "content": html_test},
    ]
    template = env.get_template(name="report_main.j2")
    output_html = template.render(
        tabs=tabs,
        test = "available" if html_test != "" else "",
        title=f"EDA Report {config_data.project_name}",
        current_time=datetime.now(),
    )
    with open(
        file=config_data.report_template, mode="w", encoding="utf-8"
    ) as f:
        f.write(output_html)
