import pandas as pd
import numpy as np
import plotly.express as px
from jinja2 import Environment, FileSystemLoader
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder

# Sample data
np.random.seed(0)
df = pd.DataFrame(
    {
        "height": np.random.normal(170, 10, 100),
        "weight": np.random.normal(65, 12, 100),
    }
)


def get_frequency_table(df, column_name) -> str:
    frequency_data = df[column_name].value_counts().to_dict()
    if set(frequency_data.keys()) != {np.False_, np.True_}:
        frequency_data = {
            f'"{str(key)}"': value for key, value in frequency_data.items()
        }
    frequency_data["Missing values"] = df[column_name].isna().sum()
    total = len(df)

    frequency_table = """
    <table class="frequency-table" border="1">
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


def get_html_from_template(template_file, context, plots=[]) -> str:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template(template_file)
    rendered_html = template.render(context=context, plots=plots)
    return rendered_html


def generate_relation_visuals(df, target=None, max_features=20):
    #    warnings.filterwarnings("ignore")
    df = df.copy().dropna()

    # Identify types
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Label encode categorical features temporarily
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Limit features for speed
    all_features = num_cols + cat_cols
    if len(all_features) > max_features:
        all_features = all_features[:max_features]

    # Compute correlation matrix
    corr_matrix = df[all_features].corr()

    # Plot Correlation Heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation Heatmap",
        labels={"color": "Correlation"},
        aspect="auto",
    )
    fig_corr.update_layout(width=900, height=900)
    correlation_html = fig_corr.to_html(full_html=False, include_plotlyjs=False)

    # Compute Mutual Information
    mi_scores = {}
    X_all_features = all_features.copy()
    if target in X_all_features:
        X_all_features.remove(target)
    target_relation = ""
    if target is not None and target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
        if y.nunique() <= 10:
            mi = mutual_info_classif(
                X[X_all_features], y, discrete_features="auto"
            )
        else:
            mi = mutual_info_regression(
                X[X_all_features], y, discrete_features="auto"
            )
        mi_scores = dict(zip(X_all_features, mi))

        # Plot MI bar chart
        mi_series = pd.Series(mi_scores).sort_values(ascending=False)
        fig_mi = px.bar(
            mi_series,
            x=mi_series.index,
            y=mi_series.values,
            title=f"Mutual Information with Target: {target}",
            labels={"x": "Feature", "y": "Mutual Information"},
        )
        fig_mi.update_layout(xaxis_tickangle=-45)
        target_relation = fig_mi.to_html(
            full_html=False, include_plotlyjs=False
        )
    return correlation_html, target_relation
