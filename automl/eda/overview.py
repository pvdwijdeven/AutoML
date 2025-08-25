# Standard library imports
import os
import re
from collections import defaultdict

# Third-party imports
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# Local application imports
from automl.library import infer_dtype


def add_links_to_headers(html: str, target: str) -> str:
    def replace_th(match) -> str:
        col_name = match.group(1)
        if col_name == target:
            link = f'<a href="#{col_name.replace(" ", "-")}" onclick="showTab(2)" class="feature-link">{col_name}</a>'
        else:
            link = f'<a href="#{col_name.replace(" ", "-")}" onclick="showTab(1)" class="feature-link">{col_name}</a>'
        return f"<th>{link}</th>"

    # Replace every <th>...</th> with a linked version
    return re.sub(r"<th>(.*?)</th>", replace_th, html).replace('border="1"', "")


def join_feature_links_with_linebreaks(
    feature_links: list[str], max_line_length: int = 90
) -> str:
    lines = []
    current_line = ""
    current_len = 0

    for html_link in feature_links:
        # Extract visible text from the HTML link
        # Assumes format: <a ...>VISIBLE TEXT</a>
        visible_text = html_link.split(">")[-2].split("<")[0]
        part_len = len(visible_text) + (
            2 if current_line else 0
        )  # +2 for ", " if needed

        if current_len + part_len > max_line_length:
            lines.append(current_line)
            current_line = html_link
            current_len = len(visible_text)
        else:
            if current_line:
                current_line += ", " + html_link
                current_len += part_len
            else:
                current_line = html_link
                current_len = len(visible_text)

    if current_line:
        lines.append(current_line)

    return "<br>".join(lines)


def create_overview_table(
    df: pd.DataFrame,
    target: str,
    target_type: str,
) -> str:
    def percentage(value: float) -> str:
        return f"{value:.1%}"

    current_dir = os.path.dirname(os.path.abspath(__file__))  # automl/gui
    automl_dir = os.path.dirname(current_dir)  # automl
    templates_dir = os.path.join(automl_dir, "templates")  # automl/templates
    env = Environment(loader=FileSystemLoader(templates_dir))
    env.filters["percentage"] = percentage

    # --- Collect the statistics ---
    n_rows = len(df)
    n_features = df.shape[1] - 1
    feature_cols = sorted(df.columns.difference([target]), key=str.lower)
    constant_cols = [
        col for col in feature_cols if df[col].nunique(dropna=False) == 1
    ]
    empty_cols = [col for col in feature_cols if df[col].count() == 0]
    duplicate_cols = (
        pd.util.hash_pandas_object(df.drop(columns=target), index=True)
        .duplicated()
        .sum()
    )

    duplicate_rows = (
        pd.util.hash_pandas_object(df.round(5), index=False).duplicated().sum()
    )

    empty_rows = (df.isnull().sum(axis=1) == df.shape[1]).sum()

    memory_usage = df.memory_usage(deep=True).sum()
    missing_values = df.isnull().sum().sum()
    type_to_features = defaultdict(list)

    for col in feature_cols:
        dtype = infer_dtype(df[col])
        type_to_features[dtype].append(
            f'<a href="#{col.replace(" ", "-")}" onclick="showTab(1)" class="feature-link">{col}</a>'
        )

    # Step 2: Build the HTML table

    feature_types = """
    <div class="feature-table-internal">
    <table style="border-collapse: collapse;">
    <thead>
        <tr>
        <th>Type</th>
        <th>Frequency</th>
        <th>Features</th>
        </tr>
    </thead>
    <tbody>
    """

    for dtype, features in type_to_features.items():
        # this is required, because wordwrap is not working with links for some reason.
        feature_list = join_feature_links_with_linebreaks(features)
        feature_types += f"<tr><td>{dtype}</td><td>{len(features)}</td><td>{feature_list}</td></tr>\n"

    feature_types += "</tbody>\n</table></div>"
    # feature_types = "HIERZO"
    samples_head = add_links_to_headers(
        df.head(10).to_html(
            index=False, na_rep="<N/A>", classes="table table-striped"
        ),
        target,
    )
    samples_middle = add_links_to_headers(
        df.iloc[n_rows // 2 - 5 : n_rows // 2 + 5].to_html(
            index=False, na_rep="<N/A>", classes="table table-striped"
        ),
        target,
    )
    samples_tail = add_links_to_headers(
        df.tail(10).to_html(
            index=False, na_rep="<N/A>", classes="table table-striped"
        ),
        target,
    )
    # --- Context dictionary for rendering ---
    context = {
        "num_features": n_features,
        "constant_cols": constant_cols,
        "duplicate_cols": duplicate_cols,
        "empty_cols": empty_cols,
        "feature_types": feature_types,
        "num_rows": n_rows,
        "duplicate_rows": duplicate_rows,
        "empty_rows": empty_rows,
        "memory_usage_kb": round(memory_usage / 1024, 2),
        "missing_values": missing_values,
        "total_cells": df.size,
        "samples_head": samples_head,
        "samples_middle": samples_middle,
        "samples_tail": samples_tail,
        "target": target,
        "target_type": target_type,
    }

    template = env.get_template("overview_table.j2")

    html_body = template.render(context)

    return html_body
