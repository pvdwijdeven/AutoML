import pandas as pd
import numpy as np
import plotly.express as px
from jinja2 import Environment, FileSystemLoader
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import LabelEncoder
from .feature_lib import select_features_by_missingness

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


def generate_relation_visuals(
    df, target="", max_features=100, max_samples=10000
):

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
    if all_features != len(df.columns):
        num_feats = all_features
    else:
        num_feats = 0
    # Step 7: Correlation matrix
    corr_matrix = df[all_features].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation Heatmap",
        labels={"color": "Correlation"},
        aspect="auto",
    )
    fig_corr.update_layout(
        autosize=False,
        width=1000,
        height=1000,
    )
    correlation_html = fig_corr.to_html(full_html=False, include_plotlyjs=False)

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
    mi_series = pd.Series(mi_scores).dropna().sort_values(ascending=False)
    fig_mi = px.bar(
        mi_series,
        x=mi_series.index,
        y=mi_series.values,
        title=f"Mutual Information with Target: {target}",
        labels={"x": "Feature", "y": "Mutual Information"},
    )
    fig_mi.update_layout(
        xaxis_tickangle=-45,
        autosize=False,
        width=1000,
        height=500,
    )
    target_relation = fig_mi.to_html(full_html=False, include_plotlyjs=False)

    return correlation_html, target_relation, num_feats
