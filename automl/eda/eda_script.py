# Standard library imports
import os
from typing import Optional, Union

# Third-party imports
from narwhals import col
import yaml
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel
from scipy.stats import entropy as scipy_entropy

# Local application imports
from automl.dataloader import ConfigData, OriginalData
from automl.library import todo  # only during develloping


class CategoricalInfo(BaseModel):
    frequency: dict[str, tuple[int, float]]
    mode: list[str]
    entropy: float
    cardinality: str


class StringInfo(BaseModel):
    samples: str  # TODO optional LLM interaction?


class NumericInfo(BaseModel):
    min_value: Union[int, float]
    max_value: Union[int, float]
    skewness: float
    mean: float
    std_dev: float


class ColumnInfo(BaseModel):
    is_constant: bool
    column_name: str
    current_type: str
    proposed_type: str
    number_unique: int
    perc_unique: float
    is_empty: bool
    number_missing: int
    perc_missing: float
    description: str
    suggestions: str
    default_prepro: str
    type_specific_info: Union[StringInfo, NumericInfo, CategoricalInfo]


def analyse_column(column: Series) -> ColumnInfo:

    num_unique = column.nunique()  # counts unique non-missing values
    perc_unique = num_unique / len(column) * 100
    num_missing = column.isna().sum()
    perc_missing = num_missing / len(column) * 100
    if is_numeric_dtype(arr_or_dtype=column):
        proposed_type = "numeric"
    else:
        proposed_type = "categorical" if perc_unique <= 20 else "string"

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
            value: (counts[value], freqs[value]) for value in counts.index
        }
        # Cardinality label
        if num_unique <= 5:
            cardinality_label = "Low"
        elif num_unique <= 20:
            cardinality_label = "Medium"
        else:
            cardinality_label = "High"
        type_specific_info = CategoricalInfo(
            frequency=freq_dict,
            mode=[str(object=x) for x in column.mode()],
            entropy=entropy_value,
            cardinality=cardinality_label,
        )

    column_info: ColumnInfo = ColumnInfo(
        is_constant=num_unique == 1,
        column_name=str(object=column.name),
        current_type=str(object=column.dtype),
        proposed_type=proposed_type,
        number_unique=num_unique,
        perc_unique=perc_unique,
        is_empty=num_missing == len(column),
        number_missing=num_missing,
        perc_missing=perc_missing,
        description="",
        suggestions="",  # todo
        default_prepro="",  # todo
        type_specific_info=type_specific_info,
    )
    return column_info


def analyse_columns(
    X_train: DataFrame, y_train: Optional[Series]
) -> dict[str, ColumnInfo]:
    column_info: dict[str, ColumnInfo] = {}
    for column in X_train.columns:
        column_info[column] = analyse_column(column=X_train[column])
    if y_train is not None:
        column_info["target"] = analyse_column(column=y_train)
    return column_info


def insert_descriptions(
    column_info: dict[str, ColumnInfo], config_data: ConfigData
) -> dict[str, ColumnInfo]:
    if config_data.description_file is not None:
        if os.path.isfile(config_data.description_file):
            with open(
                file=str(config_data.description_file),
                mode="r",
                encoding="utf-8",
            ) as file:
                descriptions: dict[str, str] = yaml.safe_load(stream=file)
            for column in column_info:
                column_name = column_info[column].column_name
                column_info[column].description = descriptions.get(
                    column_name, ""
                )
            return column_info
    return column_info


def perform_eda(config_data: ConfigData, original_data: OriginalData) -> None:

    column_info = analyse_columns(
        X_train=original_data.X_train, y_train=original_data.y_train
    )
    column_info = insert_descriptions(
        column_info=column_info, config_data=config_data
    )
    # analyse_dataset()
    # analyse_relations()
    # analyse_test_data()
    # preprocess_trial()
    # create_report()
    todo()
    return
    return column_info
