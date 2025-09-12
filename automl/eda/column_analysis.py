# Standard library imports
import os
from dataclasses import dataclass
from typing import Optional, Union

# Third-party imports
import yaml
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype
from scipy.stats import entropy as scipy_entropy

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


def analyse_column(
    column: Series, is_target: bool, cardinality_max: int = 15
) -> tuple[ColumnInfo, ColumnPlot]:
    column_plot: ColumnPlot =  ColumnPlot(feature="", target="", relation="")
    num_unique = column.nunique()
    perc_unique = num_unique / len(column) * 100
    num_missing = column.isna().sum()
    perc_missing = num_missing / len(column) * 100
    if is_numeric_dtype(arr_or_dtype=column):
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
        type_specific_info = NumericInfo(
            min_value=column.min(),
            max_value=column.max(),
            skewness=skewness,
            mean=column.mean(),
            std_dev=column.std(),
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
                )
            return column_info
    return column_info
