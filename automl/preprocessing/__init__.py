# automl/preprocessing/__init__.py

from .preprocess_old import AutoML_Preprocess_old
from .preprocessing import AutoML_Preprocess
from .general import (
    drop_duplicate_rows,
    drop_duplicate_columns,
    drop_constant_columns,
    drop_strings,
)
from .outliers import (
    skip_outliers,
    outlier_imputation_order,
    handle_outliers,
)
from .encoding import auto_encode_features, encode_target
from .missing import handle_missing_values_num, handle_missing_values_cat
from .standardizing import normalize_columns

__all__: list[str] = [
    "AutoML_Preprocess_old",
    "AutoML_Preprocess",
    "drop_duplicate_rows",
    "drop_duplicate_columns",
    "drop_constant_columns",
    "skip_outliers",
    "outlier_imputation_order",
    "handle_outliers",
    "handle_missing_values_cat",
    "handle_missing_values_num",
    "auto_encode_features",
    "encode_target",
    "drop_strings",
    "normalize_columns",
]
