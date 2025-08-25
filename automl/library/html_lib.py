# Standard library imports
import os
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
import plotly.express as px
from jinja2 import Environment, FileSystemLoader
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder

from .feature_lib import select_features_by_missingness


def get_frequency_table(df: pd.DataFrame, column_name: str) -> str:
    frequency_data = df[column_name].value_counts().to_dict()
    if set(frequency_data.keys()) != {np.False_, np.True_}:
        frequency_data = {
            f'"{str(key)}"': value for key, value in frequency_data.items()
        }
    frequency_data["Missing values"] = df[column_name].isna().sum()
    total = len(df)

    frequency_table = """
    <table class="frequency-table">
    <thead>
        <tr>
        <th>Value</th>
        <th>Count</th>
        <th>Percentage</th>
        </tr>
    </thead>
    <tbody>
    """
    for key, value in frequency_data.items():
        percentage = value / total * 100
        frequency_table += f"""
        <tr>
        <td>{key}</td>
        <td>{value}</td>
        <td>{percentage:.1f}%</td>
        </tr>
        """

    frequency_table += """
    </tbody>
    </table>
    """

    return frequency_table


def get_html_from_template(
    template_file: str, context: dict[str, Any], plots: list[str] = []
) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # automl/gui
    automl_dir = os.path.dirname(current_dir)  # automl
    templates_dir = os.path.join(automl_dir, "templates")  # automl/templates
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template(template_file)
    rendered_html = template.render(context=context, plots=plots)
    return rendered_html


def generate_relation_visuals(
    df, target="", max_features=50, max_samples=10000
) -> tuple[str, str, int]:

    # Step 1: Select features
    features = select_features_by_missingness(df, "")

    if target != "" and target not in features:
        features = [target] + features
    elif target != "" and target in features:
        features.remove(target)
        features = [target] + features

    df = df[features]

    # Step 2: Sample early
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    # Step 3: Drop rows with NA after sampling
    df = df.dropna()

    if target == "":
        target = None

    # Step 4: Detect types
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Step 5: Label encode categoricals
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    # Step 6: Restrict number of features
    all_features = num_cols + cat_cols
    if len(all_features) > max_features:
        all_features = all_features[:max_features]
    if len(all_features) != len(df.columns):
        num_feats = len(all_features)
    else:
        num_feats = 0
    # Step 7: Correlation matrix

    corr_matrix = df[all_features].corr()
    sorted_features = sorted(corr_matrix.columns, key=lambda x: x.lower())
    corr_matrix = corr_matrix.loc[sorted_features[::-1], sorted_features]
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation Heatmap",
        labels={"color": "Correlation"},
        aspect="auto",
    )
    fig_corr.update_layout(
        autosize=False, width=1000, height=1000, xaxis_side="top"
    )
    fig_corr.update_xaxes(side="top")
    correlation_html = fig_corr.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="update-corr-plot",
    )

    # Step 8: Mutual Information
    mi_scores = {}
    try:
        X_features = [f for f in all_features if f != target]
        if target is not None and target in df.columns:
            y = df[target]
            X = df[X_features]
            if y.nunique() <= 10:
                mi = mutual_info_classif(
                    X, y, discrete_features="auto", random_state=42
                )
            else:
                mi = mutual_info_regression(
                    X, y, discrete_features="auto", random_state=42
                )
            mi_scores = dict(zip(X_features, mi))
    except Exception as e:
        print(f"Error computing mutual information: {e}")
        mi_scores = {feature: None for feature in all_features}

    # Step 9: Plot MI
    mi_series = pd.Series(mi_scores).dropna().sort_values(ascending=True)

    # Pass the data as a dataframe for easy control
    df_mi = mi_series.reset_index()
    df_mi.columns = ["Feature", "MutualInformation"]

    fig_mi = px.bar(
        df_mi,
        x="Feature",
        y="MutualInformation",
        title=f"Mutual Information with Target: {target}",
        labels={
            "Feature": "Feature",
            "MutualInformation": "Mutual Information",
        },
    )

    # Set categoryorder to 'array' and provide categories in descending order of MI score
    fig_mi.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=df_mi["Feature"].tolist(),
            tickangle=-45,
            autorange=True,  # Not reversed to keep left→right ascending categories (highest MI on left)
        ),
        yaxis=dict(
            autorange=True,
        ),
        autosize=False,
        width=1000,
        height=500,
    )
    # fig_mi.update_yaxes(range=[0, 1])

    target_relation_html = fig_mi.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="update-mi-plot",
    )

    return correlation_html, target_relation_html, num_feats
