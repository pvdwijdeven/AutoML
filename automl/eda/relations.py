# Standard library imports
from dataclasses import dataclass, field
from typing import List

# Third-party imports
import numpy as np

# Local application imports
from automl.dataloader import OriginalData

from .plots import (
    calculate_cramers_v,
    plot_cat_cat_relation,
    plot_categorical_heatmap,
    plot_num_num_relation,
    plot_numeric_heatmap,
)


@dataclass(slots=True)
class InitialScanPlots:
    numeric_heatmap: str
    categorical_heatmap: str


@dataclass(slots=True)
class FeaturePair:
    feature_1: str
    feature_2: str
    metric: str
    value: float
    type: str


@dataclass(slots=True)
class PairwiseSummary:
    numeric_pairs: List[FeaturePair] = field(default_factory=list)
    categorical_pairs: List[FeaturePair] = field(default_factory=list)


@dataclass(slots=True)
class PairwisePlot:
    plot: str


@dataclass(slots=True)
class PairwisePlots:
    plots: dict[str, PairwisePlot] = field(default_factory=dict)


@dataclass(slots=True)
class RelationInfoMapping:
    initial_scan_plots: InitialScanPlots
    pairwise_summary: PairwiseSummary
    pairwise_plots: PairwisePlots


def generate_feature_relations_initial_scan(
    original_data: OriginalData,
) -> RelationInfoMapping:
    """
    Performs the initial scan for feature relations, generating overview heatmaps for
    numeric and categorical features.

    Args:
        original_data (OriginalData): A dataclass containing the training and
                                      test dataframes.

    Returns:
        RelationInfoMapping: A dataclass containing the HTML strings for the
                             generated plots.
    """
    X_train = original_data.X_train.copy()

    # Separate numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns

    # Generate numeric heatmap
    if not numeric_cols.empty:
        numeric_heatmap_html = plot_numeric_heatmap(df=X_train[numeric_cols])
    else:
        numeric_heatmap_html = "No numeric features found to plot."

    # Generate categorical heatmap
    if not categorical_cols.empty:
        categorical_heatmap_html = plot_categorical_heatmap(
            df=X_train[categorical_cols]
        )
    else:
        categorical_heatmap_html = "No categorical features found to plot."

    # Generate pairwise summary table (Step 2)
    pairwise_summary_data = generate_feature_relations_pairwise_summary(
        original_data=original_data
    )

    # Generate pairwise plots (Step 3)
    pairwise_plots_data = generate_feature_relations_pairwise_plots(
        original_data=original_data, pairwise_summary=pairwise_summary_data
    )

    return RelationInfoMapping(
        initial_scan_plots=InitialScanPlots(
            numeric_heatmap=numeric_heatmap_html,
            categorical_heatmap=categorical_heatmap_html,
        ),
        pairwise_summary=pairwise_summary_data,
        pairwise_plots=pairwise_plots_data,
    )


def generate_feature_relations_pairwise_summary(
    original_data: OriginalData, threshold: float = 0.7
) -> PairwiseSummary:
    """
    Generates a summary of feature pairs with high correlation or association.

    Args:
        original_data (OriginalData): A dataclass containing the training and test dataframes.
        threshold (float): The minimum absolute correlation/association value to be included
                           in the summary.

    Returns:
        PairwiseSummary: A dataclass containing lists of highly correlated numeric and
                         categorical feature pairs.
    """
    X_train = original_data.X_train.copy()
    numeric_pairs: List[FeaturePair] = []
    categorical_pairs: List[FeaturePair] = []

    numeric_cols = X_train.select_dtypes(include=np.number).columns
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns

    # --- Numeric Feature Pairs ---
    if not numeric_cols.empty and len(numeric_cols) > 1:
        corr_matrix = (
            X_train[numeric_cols]
            .corr(method="pearson", numeric_only=True)
            .abs()
        )

        # Unstack the matrix and filter for highly correlated pairs
        corr_matrix_unstacked = corr_matrix.stack()
        high_corr_pairs = corr_matrix_unstacked[    # type: ignore
            (corr_matrix_unstacked > threshold) & (corr_matrix_unstacked < 1.0)
        ].reset_index()
        high_corr_pairs.columns = ["feature_1", "feature_2", "value"]

        # Remove duplicate pairs (e.g., A-B and B-A)
        unique_pairs = high_corr_pairs.apply(
            lambda row: tuple(sorted((row["feature_1"], row["feature_2"]))),
            axis=1,
        ).unique()

        for pair in unique_pairs:
            value = corr_matrix.loc[pair[0], pair[1]]
            assert isinstance(value, float)
            numeric_pairs.append(
                FeaturePair(
                    feature_1=pair[0],
                    feature_2=pair[1],
                    metric="Pearson's r",
                    value=value,
                    type="numeric",
                )
            )

    # --- Categorical Feature Pairs ---
    if not categorical_cols.empty and len(categorical_cols) > 1:
        for i, col1 in enumerate(categorical_cols):
            for j in range(i + 1, len(categorical_cols)):
                col2 = categorical_cols[j]
                v_value = calculate_cramers_v(x=X_train[col1], y=X_train[col2])
                if abs(v_value) > threshold:
                    categorical_pairs.append(
                        FeaturePair(
                            feature_1=col1,
                            feature_2=col2,
                            metric="Cramer's V",
                            value=v_value,
                            type="categorical",
                        )
                    )

    return PairwiseSummary(
        numeric_pairs=numeric_pairs, categorical_pairs=categorical_pairs
    )


def generate_feature_relations_pairwise_plots(
    original_data: OriginalData, pairwise_summary: PairwiseSummary
) -> PairwisePlots:
    """
    Generates detailed plots for highly correlated feature pairs based on the summary table.

    Args:
        original_data (OriginalData): A dataclass containing the training and test dataframes.
        pairwise_summary (PairwiseSummary): A dataclass containing the lists of
                                            highly correlated feature pairs.

    Returns:
        PairwisePlots: A dataclass containing a dictionary of HTML strings for the
                       generated plots, keyed by a unique identifier for each pair.
    """
    X_train = original_data.X_train.copy()
    plots_dict = {}

    # Plot numeric pairs
    for pair in pairwise_summary.numeric_pairs:
        feature_1_series = X_train[pair.feature_1]
        feature_2_series = X_train[pair.feature_2]
        plot_html = plot_num_num_relation(
            feature_series=feature_1_series, target_series=feature_2_series
        )
        plots_dict[f"{pair.feature_1}_{pair.feature_2}"] = PairwisePlot(
            plot=plot_html
        )

    # Plot categorical pairs
    for pair in pairwise_summary.categorical_pairs:
        feature_1_series = X_train[pair.feature_1]
        feature_2_series = X_train[pair.feature_2]
        plot_html = plot_cat_cat_relation(
            feature_series=feature_1_series, target_series=feature_2_series
        )
        plots_dict[f"{pair.feature_1}_{pair.feature_2}"] = PairwisePlot(
            plot=plot_html
        )

    # Note: If your summary table were to include num-cat relations, you would add
    # a similar loop here that calls plot_num_cat_relation or plot_cat_num_relation.

    return PairwisePlots(plots=plots_dict)
