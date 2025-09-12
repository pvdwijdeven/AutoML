# Standard library imports
import base64
import io

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import numpy as np


def fig_to_base64(fig, alt_text: str) -> str:
    """Convert a matplotlib figure to a base64-encoded HTML image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_base64}" alt="alt_text" class="responsive-img"/>'


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
