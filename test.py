# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

data = sns.load_dataset("penguins")["flipper_length_mm"].dropna()


def plot_numeric_distribution(series, bins="auto"):
    """
    Generate multiple distribution plots for a numerical pandas Series.

    Plots included:
    - Histogram + KDE
    - Boxplot
    - Violin plot
    - ECDF plot
    - Rug plot + KDE
    - Stripplot
    - QQ-plot (using scipy)
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    series = series.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()

    # 1. Histogram + KDE
    sns.histplot(x=series, bins=bins, kde=True, ax=axes[0])
    axes[0].set_title("Histogram + KDE")

    # 2. Boxplot
    sns.boxplot(x=series, ax=axes[1])
    axes[1].set_title("Boxplot")

    # 3. Violin plot
    sns.violinplot(x=series, ax=axes[2])
    axes[2].set_title("Violin plot")

    # 6. Stripplot
    stats.probplot(series, dist="norm", plot=plt)
    axes[3].set_title("QQ-plot vs Normal distribution")

    plt.tight_layout()
    plt.show()


# Example with random data
# np.random.seed(42)
# data = np.random.normal(loc=50, scale=10, size=200)

plot_numeric_distribution(data)
