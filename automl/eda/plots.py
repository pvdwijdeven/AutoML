# Standard library imports
import base64
import io

# Third-party imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def fig_to_base64(fig, alt_text: str) -> str:
    """Convert a matplotlib figure to a base64-encoded HTML image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_base64}" alt="alt_text" class="responsive-img"/>'


def plot_cat_cat_relation(feature_series, target_series):
    """
    Creates a grouped bar plot and a crosstab heatmap to visualize the relationship
    between a categorical feature and a categorical target.

    Args:
        feature_series (pd.Series): The categorical feature data.
        target_series (pd.Series): The categorical target data.

    Returns:
        str: A base64-encoded HTML image string of the plots.
    """
    f_color = "dodgerblue"

    # Handle high cardinality for both feature and target
    top_n = 20
    if feature_series.nunique() > top_n or target_series.nunique() > top_n:
        top_feature_cats = feature_series.value_counts().nlargest(top_n).index
        top_target_cats = target_series.value_counts().nlargest(top_n).index

        filtered_indices = feature_series[
            feature_series.isin(top_feature_cats)
        ].index
        filtered_indices = filtered_indices.intersection(
            target_series[target_series.isin(top_target_cats)].index
        )

        feature_series = feature_series.loc[filtered_indices]
        target_series = target_series.loc[filtered_indices]
        title_suffix = f" (Top {top_n} Categories)"
    else:
        title_suffix = ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # --- Grouped Bar Plot ---
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(
        {feature_series.name: feature_series, target_series.name: target_series}
    )

    sns.countplot(
        data=plot_df,
        x=feature_series.name,
        hue=target_series.name,
        ax=axes[0],
        palette="viridis",
    )
    axes[0].set_title(
        f"Grouped Bar Plot{title_suffix}", color=f_color, fontsize=14
    )
    axes[0].set_xlabel(feature_series.name, color=f_color)
    axes[0].set_ylabel("Count", color=f_color)
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].legend(title=target_series.name)

    # --- Crosstab Heatmap ---
    crosstab_df = pd.crosstab(
        index=feature_series,
        columns=target_series,
        normalize="index",  # Normalize by row to see proportions
    )

    heatmap_plot = sns.heatmap(
        crosstab_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=axes[1],
        linewidths=0.5,
        linecolor=f_color,
        cbar_kws={"label": "Proportion", "shrink": 0.75},
    )

    # Set the color of the color bar ticks and label
    cbar = getattr(heatmap_plot.collections[0], "colorbar", None)
    if cbar is not None and hasattr(cbar, "ax"):
        cbar.ax.tick_params(colors=f_color)
        cbar.set_label("Proportion", color=f_color)
    axes[1].set_title(
        f"Crosstab Heatmap (Proportions){title_suffix}",
        color=f_color,
        fontsize=14,
    )
    axes[1].set_xlabel(target_series.name, color=f_color)
    axes[1].set_ylabel(feature_series.name, color=f_color)

    # Apply consistent styling to both subplots
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(f_color)
        ax.tick_params(axis="x", colors=f_color)
        ax.tick_params(axis="y", colors=f_color)

    plt.tight_layout()
    return fig_to_base64(
        fig, alt_text="Categorical feature vs. Categorical target plots"
    )


def plot_cat_num_relation(feature_series, target_series):
    """
    Creates a box plot and a bar plot to visualize the relationship
    between a categorical feature and a numeric target.

    Args:
        categorical_series (pd.Series): The categorical feature data.
        numeric_series (pd.Series): The numeric target data.

    Returns:
        str: A base64-encoded HTML image string of the plots.
    """
    f_color = "dodgerblue"

    # Handle high cardinality: Filter to the top 20 most frequent categories
    top_n = 20
    if feature_series.nunique() > top_n:
        top_categories = feature_series.value_counts().nlargest(top_n).index
        filtered_indices = feature_series[
            feature_series.isin(top_categories)
        ].index
        feature_series = feature_series.loc[filtered_indices]
        target_series = target_series.loc[filtered_indices]
        title_suffix = f" (Top {top_n} Categories)"
    else:
        title_suffix = ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Box Plot ---
    sns.boxplot(
        x=feature_series,
        y=target_series,
        ax=axes[0],
        whiskerprops={"color": f_color},
        capprops={"color": f_color},
        flierprops=dict(
            marker="o",
            markerfacecolor="blue",
            markeredgecolor=f_color,
            markersize=5,
            alpha=0.7,
        ),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[0].set_title(f"Box Plot{title_suffix}", color=f_color, fontsize=14)
    axes[0].set_xlabel(feature_series.name, color=f_color)
    axes[0].set_ylabel(target_series.name, color=f_color)
    axes[0].tick_params(axis="x", labelrotation=45)

    # --- Bar Plot (with Mean) ---
    sns.barplot(
        x=feature_series,
        y=target_series,
        ax=axes[1],
        errorbar=None,  # Set to None to remove confidence intervals
    )
    axes[1].set_title(
        f"Bar Plot (Mean){title_suffix}", color=f_color, fontsize=14
    )
    axes[1].set_xlabel(feature_series.name, color=f_color)
    axes[1].set_ylabel(target_series.name, color=f_color)
    axes[1].tick_params(axis="x", labelrotation=45)

    # Apply consistent styling to both subplots
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(f_color)
        ax.tick_params(axis="x", colors=f_color)
        ax.tick_params(axis="y", colors=f_color)
        ax.grid(True, color=f_color, linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(
        fig, alt_text="Categorical feature vs. Numeric target plots"
    )


def plot_num_cat_relation(feature_series, target_series):
    """
    Creates a box plot and a violin plot to visualize the relationship
    between a numeric feature and a categorical target.

    Args:
        numeric_series (pd.Series): The numeric feature data.
        categorical_series (pd.Series): The categorical target data.

    Returns:
        str: A base64-encoded HTML image string of the plots.
    """
    f_color = "dodgerblue"

    # Handle high cardinality: Filter to the top 20 most frequent categories
    top_n = 20
    if target_series.nunique() > top_n:
        top_categories = target_series.value_counts().nlargest(top_n).index
        filtered_indices = target_series[
            target_series.isin(top_categories)
        ].index
        feature_series = feature_series.loc[filtered_indices]
        target_series = target_series.loc[filtered_indices]
        title_suffix = f" (Top {top_n} Categories)"
    else:
        title_suffix = ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Box Plot ---
    sns.boxplot(
        x=target_series,
        y=feature_series,
        ax=axes[0],
        whiskerprops={"color": f_color},
        capprops={"color": f_color},
        flierprops=dict(
            marker="o",
            markerfacecolor="blue",
            markeredgecolor=f_color,
            markersize=5,
            alpha=0.7,
        ),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[0].set_title(f"Box Plot{title_suffix}", color=f_color, fontsize=14)
    axes[0].set_xlabel(target_series.name, color=f_color)
    axes[0].set_ylabel(feature_series.name, color=f_color)
    axes[0].tick_params(axis="x", labelrotation=45)

    # --- Violin Plot ---
    sns.violinplot(
        x=target_series, y=feature_series, ax=axes[1], inner="quartile"
    )
    axes[1].set_title(f"Violin Plot{title_suffix}", color=f_color, fontsize=14)
    axes[1].set_xlabel(target_series.name, color=f_color)
    axes[1].set_ylabel(feature_series.name, color=f_color)
    axes[1].tick_params(axis="x", labelrotation=45)

    # Apply consistent styling to both subplots
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(f_color)
        ax.tick_params(axis="x", colors=f_color)
        ax.tick_params(axis="y", colors=f_color)
        ax.grid(True, color=f_color, linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(
        fig, alt_text="Numeric feature vs. Categorical target plots"
    )


def plot_num_num_relation(feature_series, target_series):
    """
    Creates a scatter plot and a hexbin plot to visualize the relationship
    between a numeric feature and a numeric target.

    Args:
        feature_series (pd.Series): The numeric feature data.
        target_series (pd.Series): The numeric target data.

    Returns:
        str: A base64-encoded HTML image string of the plots.
    """

    # Get the base colormap
    base_cmap = plt.cm.get_cmap("Blues")

    # Make a new colormap with alpha fading in
    colors = base_cmap(np.linspace(0, 1, 256))
    alphas = np.linspace(0, 1, 256)  # alpha goes from 0 (transparent) to 1 (opaque)
    colors[:, -1] = alphas           # replace alpha channel

    Blues_alpha = mcolors.ListedColormap(colors, name="Blues_alpha")

    f_color = "dodgerblue"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Scatter Plot ---
    axes[0].scatter(feature_series, target_series, alpha=0.5, color=f_color)
    axes[0].set_title("Scatter Plot", color=f_color, fontsize=14)
    axes[0].set_xlabel(feature_series.name, color=f_color)
    axes[0].set_ylabel(target_series.name, color=f_color)

    # --- Hexbin Plot ---
    axes[1].hexbin(feature_series, target_series, gridsize=30, cmap=Blues_alpha)
    axes[1].set_title("Hexbin Plot", color=f_color, fontsize=14)
    axes[1].set_xlabel(feature_series.name, color=f_color)
    axes[1].set_ylabel(target_series.name, color=f_color)
    cb = fig.colorbar(axes[1].collections[0], ax=axes[1])
    cb.set_label("count", color=f_color)

    # Apply consistent styling to both subplots
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(f_color)
        ax.tick_params(axis="x", colors=f_color, labelrotation=45)
        ax.tick_params(axis="y", colors=f_color)
        ax.grid(True, color=f_color, linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(
        fig, alt_text="Numeric feature vs. Numeric target plots"
    )


def plot_feature_importance(sorted_importance: dict[str, float]) -> str:
    """
    Creates a horizontal bar plot of feature importance scores.

    Args:
        sorted_importance (dict[str, float]): A dictionary mapping feature names
                                            to their importance score, sorted
                                            in descending order.

    Returns:
        str: A base64-encoded HTML image string of the plot.
    """
    # Create a DataFrame from the sorted importance dictionary
    df_importance = pd.DataFrame(
        list(sorted_importance.items()), columns=["feature", "importance"]
    )

    # Convert importance scores to a percentage scale for better readability
    total_importance = df_importance["importance"].sum()
    if total_importance > 0:
        df_importance["importance_percentage"] = (
            df_importance["importance"] / total_importance
        ) * 100
    else:
        df_importance["importance_percentage"] = 0

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, len(df_importance) * 0.4 + 2))
    f_color = "dodgerblue"

    # Use Seaborn to create the bar plot
    sns.barplot(
        x="importance_percentage",
        y="feature",
        data=df_importance,
        ax=ax,
    )

    # Style the plot to match the rest of the EDA report
    ax.set_title(
        "Feature Importance (Mutual Information)", color=f_color, fontsize=16
    )
    ax.set_xlabel("Relative Importance (%)", color=f_color, fontsize=12)
    ax.set_ylabel("Feature", color=f_color, fontsize=12)
    ax.tick_params(axis="x", colors=f_color)
    ax.tick_params(axis="y", colors=f_color)

    # Set spines to blue
    for spine in ax.spines.values():
        spine.set_color(f_color)

    # Add a grid for better readability
    ax.grid(axis="x", color=f_color, linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Convert the plot to a base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return f'<img src="data:image/png;base64,{img_base64}" alt="Feature Importance Plot" class="responsive-img"/>'


def plot_numeric_distribution(series, bins=20):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    series = series.dropna()
    f_color = "dodgerblue"
    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    axes = axes.flatten()

    # 1. Histogram + KDE
    # If the series is integer type, create bins aligned to integers
    if pd.api.types.is_integer_dtype(series):
        min_val = series.min()
        max_val = series.max()
        bins = np.linspace(min_val - 0.5, max_val + 0.5, bins + 1)
        bw_adjust = 1
    else:
        bw_adjust = 1
    sns.histplot(
        x=series, bins=bins, kde=True, ax=axes[0], kde_kws={"bw_adjust": bw_adjust}
    )
    # Force x-axis to show only integer ticks
    # if pd.api.types.is_integer_dtype(series):
    #     axes[0].set_xticks(np.arange(series.min(), series.max() + 1))
    axes[0].tick_params(axis="x", rotation=45, colors=f_color)
    axes[0].tick_params(axis="y", colors=f_color)
    axes[0].set_title("Histogram + KDE")

    # 2. Boxplot
    sns.boxplot(
        x=series,
        ax=axes[1],
        whiskerprops={"color": f_color, "linewidth": 2},
        capprops={"color": f_color, "linewidth": 2},
        flierprops=dict(
            marker="o",
            markerfacecolor="blue",
            markeredgecolor=f_color,
            markersize=8,
            alpha=0.7,
        ),
        medianprops=dict(color=f_color, linewidth=2),
    )
    axes[1].set_title("Boxplot")

    # 3. Violin plot
    sns.violinplot(x=series, ax=axes[2])
    axes[2].set_title("Violin plot")

    # 6. Stripplot
    stats.probplot(series, dist="norm", plot=plt)
    axes[3].set_title("QQ-plot vs Normal distribution")

    for ax in axes:
        # Set spines and tick params to blue
        for spine in ax.spines.values():
            spine.set_color(f_color)
        ax.tick_params(axis="x", colors=f_color, labelrotation=45)
        ax.tick_params(axis="y", colors=f_color)
        # Set labels (if existing) to blue
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), color=f_color)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), color=f_color)
        # Set grid to blue
        ax.grid(True, color=f_color, linestyle="--", linewidth=0.5, alpha=0.3)
        # Set title to blue (just in case)
        ax.title.set_color(f_color)

    plt.tight_layout()
    # plt.show()
    return fig_to_base64(fig, alt_text="Numeric distribution plots")


def plot_categorical_distribution(series, bins=20):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    series = series.dropna()

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    sns.countplot(x=series)
    f_color = "dodgerblue"
    axes.set_title("countplot")
    # Set spines and tick params to blue
    for spine in axes.spines.values():
        spine.set_color(f_color)
    axes.tick_params(axis="x", colors=f_color, labelrotation=45)
    axes.tick_params(axis="y", colors=f_color)
    # Set labels (if existing) to blue
    if axes.get_xlabel():
        axes.set_xlabel(axes.get_xlabel(), color=f_color)
    if axes.get_ylabel():
        axes.set_ylabel(axes.get_ylabel(), color=f_color)
    # Set grid to blue
    axes.grid(True, color=f_color, linestyle="--", linewidth=0.5, alpha=0.3)
    # Set title to blue (just in case)
    axes.title.set_color(f_color)
    plt.tight_layout()
    # plt.show()
    return fig_to_base64(fig, alt_text="Categorical distribution plot")
