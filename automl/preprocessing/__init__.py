# automl/preprocessing/__init__.py

from .preprocess_old import AutoML_Preprocess_old
from .preprocessing import AutoML_Preprocess
from .general import (
    drop_duplicate_rows,
    drop_duplicate_columns,
    drop_constant_columns,
)
from .outliers import (
    skip_outliers,
    decide_outlier_imputation_order,
    handle_outliers,
)

__all__: list[str] = [
    "AutoML_Preprocess_old",
    "AutoML_Preprocess",
    "drop_duplicate_rows",
    "drop_duplicate_columns",
    "drop_constant_columns",
    "skip_outliers",
    "decide_outlier_imputation_order",
    "handle_outliers",
]
