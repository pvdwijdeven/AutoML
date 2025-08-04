import pandas as pd
import plotly.express as px
from automl_libs import infer_dtype
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

import matplotlib.colors as mcolors


def missing_data_summary(df):
    # Count of missing values per column
    missing_count = df.isnull().sum()

    # Percentage of missing values per column
    missing_percentage = (missing_count / len(df)) * 100
    mask = missing_percentage > 0
    missing_count = missing_count[mask]
    missing_percentage = missing_percentage[mask]
    # Combine into a DataFrame and sort descending
    column_info = pd.DataFrame(
        {
            "Missing Count": missing_count.astype(str) + " of " + str(len(df)),
            "Missing Percentage": missing_percentage,
        }
    ).sort_values(by="Missing Percentage", ascending=False)
    column_info["Missing Percentage"] = column_info["Missing Percentage"].map(
        "{:.2f}%".format
    )
    # General missing data statistics
    total_missing = missing_count.sum()
    total_missing_text = f"{total_missing} of {len(df) * len(df.columns)} ({total_missing / (len(df) * len(df.columns)):.2%})"
    rows_with_missing = df.isnull().any(axis=1).sum()
    rows_with_missing_text = (
        f"{rows_with_missing} of {len(df)} ({rows_with_missing / len(df):.2%})"
    )
    columns_something_missing = (missing_count > 0).sum()
    columns_none_missing = (missing_count == 0).sum()
    columns_something_missing_text = f"{columns_something_missing} of {len(df.columns)} ({columns_something_missing / len(df.columns):.2%})"
    columns_none_missing_text = f"{columns_none_missing} of {len(df.columns)} ({columns_none_missing / len(df.columns):.2%})"
    columns_all_missing = (missing_count == len(df)).sum()
    columns_all_missing_text = f"{columns_all_missing} of {len(df.columns)} ({columns_all_missing / len(df.columns):.2%})"

    general_info = pd.DataFrame(
        {
            "Metric": [
                "Total Missing Values",
                "Rows with Missing Values",
                "Columns with 100% Missing",
                "Columns with something Missing",
                "Columns with nothing Missing",
            ],
            "Value": [
                total_missing_text,
                rows_with_missing_text,
                columns_all_missing_text,
                columns_something_missing_text,
                columns_none_missing_text,
            ],
        }
    )

    # Convert to HTML
    column_info_html = column_info.to_html(classes=["frequency-table"])
    general_info_html = general_info.to_html(header=False, index=False)

    return column_info_html, general_info_html


def plot_missingness_matrix(df, top_n=100) -> str:
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if missing_counts.empty:
        return "<p>No columns with missing values to analyze.</p>"

    top_missing_cols = (
        missing_counts.sort_values(ascending=False).head(top_n).index
    )
    mask = (
        df[top_missing_cols].isnull().astype(int).T
    )  # transpose for features on Y

    if mask.values.max() == mask.values.min():
        return "<p>Only one value (all missing or all present) — can't render heatmap.</p>"

    # Colormap: 0 = blue, 1 = white
    cmap = mcolors.ListedColormap(["#1f77b4", "#ffffff"])
    bounds = [0, 0.5, 1]  # boundaries between 0 and 1
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(
        figsize=(min(20, mask.shape[1] * 0.15), 1 + mask.shape[0] * 0.4)
    )
    cax = ax.imshow(mask, aspect="auto", cmap=cmap, norm=norm)

    ax.set_title("Missing Value Matrix")
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Feature")

    ax.set_yticks(np.arange(mask.shape[0]))
    ax.set_yticklabels(mask.index.tolist())
    ax.set_xticks([])  # skip xticks for simplicity

    plt.tight_layout()

    # Export as base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;" />'


def plot_missing_correlation(df, top_n=100):

    # Step 1: Create missing value indicator
    missing_df = df.isnull().astype(int)

    # Step 2: Remove columns with 0 missing values
    missing_df = missing_df.loc[:, missing_df.sum() > 0]
    if missing_df.empty:
        return "No columns with missing values to analyze."
    # Step 3: Focus on top N features with most missing values
    top_cols = missing_df.sum().sort_values(ascending=False).head(top_n).index
    missing_df = missing_df[top_cols]

    # Step 4: Correlation matrix of missing patterns
    corr = missing_df.corr()

    # Step 4: Plot with Plotly
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation of Missingness",
    )
    fig.update_layout(
        xaxis_title="Feature", yaxis_title="Feature", height=600, width=600
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        default_width="100%",
        div_id="my-plot2",
        config=dict(responsive=True),
    )


def generate_missing_summary(df, drop_col_thresh=0.6, drop_row_thresh=0.05):
    """
    Suggest how to handle missing values for each feature in a DataFrame,
    including imputation strategies even if imputation is not strictly recommended.

    Parameters:
    - df: pandas DataFrame
    - drop_col_thresh: float, threshold for dropping a column (default 60%)
    - drop_row_thresh: float, threshold for dropping rows (default 5%)

    Returns:
    - HTML table as a string
    """

    n_rows = len(df)
    results = []

    for col in df.columns:
        n_missing = df[col].isnull().sum()
        pct_missing = n_missing / n_rows

        if n_missing == 0:
            continue  # skip columns without missing values

        col_type = infer_dtype(df[col])
        n_unique = df[col].nunique(dropna=True)
        skewness = (
            df[col].skew() if pd.api.types.is_numeric_dtype(df[col]) else None
        )

        # Default values
        suggestion = ""
        strategy = ""

        # 1. Drop column if mostly missing
        if pct_missing > drop_col_thresh:
            suggestion = "Drop column"
            strategy = "Alternatively: "

        # 2. Drop rows if only few missing
        elif pct_missing < drop_row_thresh:
            suggestion = "Drop rows with missing"
            # Still recommend a strategy in case user prefers imputation
            strategy = "Alternatively: "
        else:
            suggestion = "Impute"

        # 3. Recommend imputation strategy (always, per your request)
        if col_type in ["integer", "floating", "mixed-integer-float"]:
            if skewness is not None and abs(skewness) > 1:
                strategy += "median imputation (skewed)"
            else:
                strategy += "mean imputation"
            if pct_missing > 0.2:
                strategy += " or KNN/iterative imputation"
        elif col_type in ["object", "string", "categorical", "boolean"]:
            impute_options = []

            if n_unique <= 10:
                impute_options.append("mode imputation")
            if 10 < n_unique <= 50:
                impute_options.append("'Unknown' label")
            if n_unique > 10:
                impute_options.append("frequency or target encoding")

            strategy += " or ".join(impute_options)

            if pct_missing > 0.2:
                strategy += "; consider model-based imputation if valuable"
        elif "datetime" in col_type:
            strategy += "most frequent date or interpolation"
        else:

            strategy += "custom strategy required (complex type)"

        # 4. Consider group-based imputation advice
        if (
            suggestion == "Impute"
            and col_type in ["integer", "floating", "object", "category"]
            and n_unique > 1
        ):
            strategy += "; optionally use group-based imputation if related feature available"

        results.append(
            {
                "Column": col,
                "Missing Count": n_missing,
                "Missing %": f"{pct_missing:.1%}",
                "Type": str(col_type),
                "Suggestion": suggestion,
                "Imputation Strategy": strategy,
            }
        )
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(
            by="Missing Count",
            ascending=False,
        )
    return df_results.to_html(
        classes=["frequency-table"], index=False, justify="left"
    )
