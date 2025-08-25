# Standard library imports
import base64
import io

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

from .scoring import sort_ascending


def fig_to_base64(fig, alt_text: str) -> str:
    """Convert a matplotlib figure to a base64-encoded HTML image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_base64}" alt="{alt_text}" class="responsive-img" style="display: block; margin: 0 auto; width: 50%;"/>'


def plot_models_step1(meta_data, step) -> str:
    # Extract model metrics and sort by mean score (descending)
    models_data = []

    # Extract all mean_scores first
    all_scores = [
        float(metrics["mean_score"]) for metrics in meta_data[step].values()
    ]

    mean = np.mean(all_scores)
    std = np.std(all_scores)

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    for model, metrics in meta_data[step].items():
        mean_score = float(metrics["mean_score"])
        if lower_bound <= mean_score <= upper_bound:
            models_data.append(
                (
                    model,
                    mean_score,
                    float(metrics["std_score"]),
                    float(metrics["time_taken"]) / 5,
                )
            )

    # Sort by mean score (descending)
    models_data.sort(
        key=lambda x: x[1],
        reverse=not sort_ascending(scorer_name=meta_data["scoring"]),
    )

    # Unzip sorted data
    models, scoring_means, scoring_stds, times = zip(*models_data)

    x = np.arange(len(models))
    width = 0.5

    fig, ax1 = plt.subplots()

    # Plot mean accuracy bars
    ax1.bar(
        x,
        scoring_means,
        width,
        label=meta_data["scoring"],
        alpha=0.6,
        color="blue",
    )

    # Plot std bars centered on top of mean bars
    bottom_std = [m - s / 2 for m, s in zip(scoring_means, scoring_stds)]
    ax1.bar(
        x,
        scoring_stds,
        width,
        alpha=0.4,
        color="red",
        bottom=bottom_std,
        label="Std Dev",
    )

    ax1.set_ylabel(meta_data["scoring"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")  # << rotated labels
    # ax1.set_ylim(0, 1)

    # Create second y-axis for time
    ax2 = ax1.twinx()
    ax2.plot(x, times, "o-", color="green", label="Time (s)")
    ax2.set_ylabel("Time (seconds)")

    # Combine legends from both axes
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(
        lines,
        labels,
        loc="upper right",
        fontsize=8,
        labelspacing=0.3,
        handlelength=1.5,
    )

    plt.title(f"Model {meta_data["scoring"]}, Std Dev and Time")
    plot_html = fig_to_base64(fig=fig, alt_text=f"step{step}")
    return plot_html


def plot_models_step2(rows_step2) -> str:
    # Extract data tuples: (model, best_score, , time)

    models_data = []
    for row in rows_step2:
        model = row["model"]
        best_score = float(row["best_score"])
        time_taken = float(row["time_taken"])
        models_data.append((model, best_score, time_taken))

    # Sort models by best_score descending
    models_data.sort(key=lambda x: x[1], reverse=True)

    # Unpack sorted data
    models, scores, times = zip(*models_data)

    x = np.arange(len(models))
    width = 0.5

    fig, ax1 = plt.subplots()

    # Plot bars for scores
    ax1.bar(x, scores, width, label="Score", alpha=0.6, color="blue")
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")

    # Create second y axis for time
    ax2 = ax1.twinx()
    ax2.plot(x, times, "o-", color="green", label="Time (s)")
    ax2.set_ylabel("Time (seconds)")

    # Combine legends from both axes
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(
        lines,
        labels,
        loc="upper right",
        fontsize=8,
        labelspacing=0.3,
        handlelength=1.5,
    )

    plt.title("Model Score and Time")

    plot_html = fig_to_base64(fig=fig, alt_text="step2")
    return plot_html
