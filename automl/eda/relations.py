# Standard library imports
from dataclasses import asdict, dataclass
from typing import Any, Union

# Third-party imports
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

# Local application imports
from automl.dataloader import OriginalData
from automl.library import todo  # only during develloping
from .plots import plot_numeric_heatmap, plot_categorical_heatmap


@dataclass
class InitialScanPlots:
    numeric_heatmap: str
    categorical_heatmap: str


@dataclass
class RelationInfoMapping:
    initial_scan_plots: InitialScanPlots
    # Add other dataclasses for later steps here, like:
    # pairwise_summary_table: PairwiseSummary
    # pairwise_plots: PairwisePlots


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

    return RelationInfoMapping(
        initial_scan_plots=InitialScanPlots(
            numeric_heatmap=numeric_heatmap_html,
            categorical_heatmap=categorical_heatmap_html,
        )
    )
