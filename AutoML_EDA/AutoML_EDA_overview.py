from jinja2 import Environment, FileSystemLoader
from AutoML_Libs import infer_dtype
from collections import Counter


def create_overview_table(df) -> str:
    def percentage(value) -> str:
        return f"{value:.1%}"

    env = Environment(loader=FileSystemLoader("templates"))
    env.filters["percentage"] = percentage

    # --- Collect the statistics ---
    n_rows = len(df)
    n_features = df.shape[1]

    constant_cols = [
        col for col in df.columns if df[col].nunique(dropna=False) == 1
    ]
    empty_cols = [col for col in df.columns if df[col].count() == 0]
    duplicate_cols = df.T.duplicated().sum()

    duplicate_rows = df.duplicated().sum()
    empty_rows = (df.isnull().sum(axis=1) == df.shape[1]).sum()

    memory_usage = df.memory_usage(deep=True).sum()
    missing_values = df.isnull().sum().sum()

    feature_types = """
    <table border="1" style="border-collapse: collapse;">
    <thead>
        <tr>
        <th>Type</th>
        <th>Frequency</th>
        </tr>
    </thead>
    <tbody>
    """
    for dtype, count in Counter(
        infer_dtype(df[col]) for col in df.columns
    ).items():
        feature_types += f"    <tr><td>{dtype}</td><td>{count}</td></tr>\n"

    feature_types += "  </tbody>\n</table>"

    # Samples
    samples_head = df.head(10).to_html(index=False)
    samples_middle = df.iloc[n_rows // 2 - 5 : n_rows // 2 + 5].to_html(
        index=False
    )
    samples_tail = df.tail(10).to_html(index=False)

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
    }

    template = env.get_template("overview_table.html")
    html_body = template.render(context)

    return html_body
