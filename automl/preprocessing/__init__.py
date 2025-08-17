# automl/preprocessing/__init__.py

from .preprocessing import preprocess
from .general import (
    drop_duplicate_rows,
    drop_duplicate_columns,
    drop_constant_columns,
    drop_strings,
    skip_outliers,
    detect_dataset_type,
)
from .outliers import (
    outlier_imputation_order,
    handle_outliers,
)
from .encoding import auto_encode_features
from .missing import handle_missing_values
from .standardizing import (
    normalize_columns,
)
from .target import (
    is_target_categorical,
    TargetTransformer,
    encode_target,
    decode_target,
    standardize_target,
)

from .transformer import AutomlTransformer

__all__: list[str] = [
    "preprocess",
    "is_target_categorical",
    "decode_target",
    "drop_duplicate_rows",
    "drop_duplicate_columns",
    "drop_constant_columns",
    "skip_outliers",
    "outlier_imputation_order",
    "handle_outliers",
    "handle_missing_values",
    "auto_encode_features",
    "encode_target",
    "drop_strings",
    "normalize_columns",
    "standardize_target",
    "TargetTransformer",
    "detect_dataset_type",
    "AutomlTransformer",
]
