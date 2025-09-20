# Standard library imports
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame, Series
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis, shapiro

# Local application imports
from automl.dataloader import ConfigData

from .plots import plot_categorical_distribution, plot_numeric_distribution


@dataclass
class ColumnPlot:
    feature: str
    target: str
    relation: str


@dataclass
class ColumnPlotMapping:
    columnplot: dict[str, ColumnPlot]


@dataclass
class CategoricalInfo:
    frequency: Union[dict[str, tuple[int, float]], str]
    mode: str
    entropy: float
    cardinality: str


@dataclass
class StringInfo:
    samples: str


@dataclass
class NumericInfo:
    min_value: Union[int, float]
    max_value: Union[int, float]
    skewness: float
    mean: float
    std_dev: float
    outliers_count: int
    extreme_outliers_count: int
    lower_bound: Union[int, float]
    upper_bound: Union[int, float]
    extreme_lower: Union[int, float]
    extreme_upper: Union[int, float]
    threshold_method: str
    kurtosis: Union[float, str]
    prop_zero: float


@dataclass
class ColumnInfo:
    is_target: bool
    is_constant: bool
    column_name: str
    current_type: str
    proposed_type: str
    number_unique: int
    perc_unique: float
    is_empty: bool
    number_missing: int
    perc_missing: float
    duplicate_columns: str
    description: str
    suggestions: str
    default_prepro: str
    type_specific_info: Union[StringInfo, NumericInfo, CategoricalInfo]


@dataclass
class ColumnInfoMapping:
    columninfo: dict[str, ColumnInfo]

    # Convenience dict-like behavior
    def __getitem__(self, item: str) -> ColumnInfo:
        return self.columninfo[item]

    # Use a custom method to iterate over keys to avoid BaseModel __iter__ override issues
    def iter_keys(self):
        return iter(self.columninfo)

    def items(self):
        return self.columninfo.items()

    def keys(self):
        return self.columninfo.keys()

    def values(self):
        return self.columninfo.values()


def get_threshold_method(column: pd.Series, max_sample_size: int = 5000) -> str:
    data = column.dropna()
    # if low cardinality or many zeros, use empirical
    n_unique = data.nunique(dropna=True)
    prop_zero = (data == 0).sum() / max(1, len(data))
    if len(data) < 3 or len(data) > max_sample_size:
        return "iqr"
    if n_unique <= 10 or prop_zero > 0.3:
        # low-cardinality or zero-inflated: use empirical rules
        return "empirical"
    _, p_value = shapiro(data)  # still ok for moderate sizes
    return "zscore" if p_value > 0.05 else "iqr"


def get_outliers(
    column: pd.Series,
    extreme_outlier_factor: float = 2.0,
    empirical_quantile: float = 0.99,
) -> dict[str, Any]:
    column = column.copy()
    threshold_method = get_threshold_method(column)

    if threshold_method == "iqr":
        Q1, Q3 = column.quantile(0.25), column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        extreme_lower = Q1 - extreme_outlier_factor * 1.5 * IQR
        extreme_upper = Q3 + extreme_outlier_factor * 1.5 * IQR

    elif threshold_method == "empirical":
        # Focus on non-zero behaviour but keep domain limits
        nonzero = column[column > 0]
        if len(nonzero) == 0:
            # all zeros -> no outliers
            lower_bound = upper_bound = extreme_lower = extreme_upper = 0
        else:
            # use an upper quantile of the empirical distribution of values (>0)
            upper_q = int(np.ceil(nonzero.quantile(empirical_quantile)))
            # make sure we don't exceed possible domain
            domain_max = (
                column.max()
                if pd.api.types.is_numeric_dtype(column)
                else upper_q
            )
            upper_bound = min(upper_q, domain_max)
            lower_bound = 0  # zeros are common, so lower bound stays 0
            # extreme bounds could be just the max possible observed value
            extreme_upper = column.max()
            extreme_lower = 0

        # If you prefer cumulative-frequency-based threshold:
        # compute cumulative frequency and pick smallest v with cumfreq >= 0.99

    else:  # zscore
        mean, std = column.mean(), column.std(ddof=0)
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        extreme_lower = mean - extreme_outlier_factor * 3 * std
        extreme_upper = mean + extreme_outlier_factor * 3 * std

    outliers_mask = (column < lower_bound) | (column > upper_bound)
    extreme_mask = (column < extreme_lower) | (column > extreme_upper)

    result = {
        "outliers_count": int(outliers_mask.sum()),
        "extreme_outliers_count": int(extreme_mask.sum()),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "extreme_lower": extreme_lower,
        "extreme_upper": extreme_upper,
        "threshold_method": threshold_method,
        "kurtosis": (
            float(kurtosis(column.dropna(), fisher=False))
            if len(column.dropna()) > 0
            else ""
        ),
        "prop_zero": float((column == 0).sum() / max(1, len(column))),
    }
    return result


def analyse_column(
    column: Series, is_target: bool, cardinality_max: int = 15
) -> tuple[ColumnInfo, ColumnPlot]:
    column_plot: ColumnPlot = ColumnPlot(feature="", target="", relation="")
    num_unique = column.nunique()
    perc_unique = num_unique / len(column) * 100
    num_missing = column.isna().sum()
    perc_missing = num_missing / len(column) * 100
    if pd.api.types.is_numeric_dtype(arr_or_dtype=column) and column.nunique()>2:
        proposed_type = "numeric"
        column_plot.feature = plot_numeric_distribution(series=column)
    else:
        proposed_type = (
            "categorical" if perc_unique <= cardinality_max else "string"
        )

    if proposed_type == "numeric":
        skewness = column.skew()
        skewness = (
            float(skewness)
            if isinstance(skewness, float) or isinstance(skewness, int)
            else 0
        )
        dict_outlier = get_outliers(column=column)
        type_specific_info = NumericInfo(
            min_value=column.min(),
            max_value=column.max(),
            skewness=skewness,
            mean=column.mean(),
            std_dev=column.std(),
            outliers_count=dict_outlier["outliers_count"],
            extreme_outliers_count=dict_outlier["extreme_outliers_count"],
            lower_bound=dict_outlier["lower_bound"],
            upper_bound=dict_outlier["upper_bound"],
            extreme_lower=dict_outlier["extreme_lower"],
            extreme_upper=dict_outlier["extreme_upper"],
            threshold_method=dict_outlier["threshold_method"],
            kurtosis=dict_outlier["kurtosis"],
            prop_zero=dict_outlier["prop_zero"],
        )
    elif proposed_type == "string":
        type_specific_info = StringInfo(
            samples=", ".join(column.value_counts().head(5).index.tolist()),
        )
    else:
        column_plot.feature = plot_categorical_distribution(series=column)
        value_counts = column.value_counts(normalize=True, dropna=True)
        entropy_value = float(
            scipy_entropy(value_counts, base=2)
        )  # base-2: bits
        # Count occurrences
        counts = column.value_counts(sort=False)

        # Calculate frequency (proportion)
        freqs = column.value_counts(normalize=True, sort=False)

        # Combine into dictionary
        freq_dict = {
            str(object=value): (counts[value], freqs[value])
            for value in counts.index
        }
        # Cardinality label
        if num_unique <= 5:
            cardinality_label = "Low"
        elif num_unique <= cardinality_max:
            cardinality_label = "Medium"
        else:
            cardinality_label = "High"
        type_specific_info = CategoricalInfo(
            frequency=freq_dict,
            mode=", ".join([str(object=x) for x in column.mode()]),
            entropy=entropy_value,
            cardinality=cardinality_label,
        )
    column_info: ColumnInfo = ColumnInfo(
        is_target=is_target,
        is_constant=num_unique == 1,
        column_name=str(object=column.name),
        current_type=str(object=column.dtype),
        proposed_type=proposed_type,
        number_unique=num_unique,
        perc_unique=perc_unique,
        is_empty=num_missing == len(column),
        number_missing=num_missing,
        perc_missing=perc_missing,
        duplicate_columns="",
        description="",
        suggestions="",  # todo via LLM?
        default_prepro="",  # todo after prepro test
        type_specific_info=type_specific_info,
    )

    return column_info, column_plot


def analyse_columns(
    X_train: DataFrame,
    dict_duplicates: dict[str, list[str]],
    y_train: Optional[Series],
) -> tuple[ColumnInfoMapping, ColumnPlotMapping]:
    column_info = {}
    plot_info = {}
    for column in X_train.columns:
        column_info[column], plot_info[column] = analyse_column(
            column=X_train[column], is_target=False
        )
        column_info[column].duplicate_columns = ", ".join(
            dict_duplicates[column]
        )
    if y_train is not None:
        column_info["target"], plot_info["target"] = analyse_column(
            column=y_train, is_target=True
        )
    return ColumnInfoMapping(**{"columninfo": column_info}), ColumnPlotMapping(
        **{"columnplot": plot_info}
    )


def insert_descriptions(
    column_info: ColumnInfoMapping, config_data: ConfigData
) -> ColumnInfoMapping:
    if config_data.description_file is not None:
        if os.path.isfile(config_data.description_file):
            with open(
                file=str(config_data.description_file),
                mode="r",
                encoding="utf-8",
            ) as file:
                descriptions: dict[str, str] = yaml.safe_load(stream=file)
            for column in column_info.keys():
                column_name = column_info[column].column_name
                column_info[column].description = descriptions.get(
                    column_name, ""
                ).replace("\n","<br>")
            return column_info
    return column_info
