from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_object_dtype,
)
import pandas as pd


def infer_dtype(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "boolean"
    if is_integer_dtype(series):
        return "integer"
    if is_float_dtype(series):
        return "float"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    if is_string_dtype(series):
        return "string"
    if is_object_dtype(series):
        return "object"
    return str(series.dtype)
