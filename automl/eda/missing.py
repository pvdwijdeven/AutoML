# Standard library imports
from dataclasses import dataclass, asdict

# Third-party imports
import numpy as np
import pandas as pd
from automl.dataloader import OriginalData
from .column_analysis import ColumnInfoMapping, NumericInfo
from .plots import plot_missingness_matrix

@dataclass
class MissingInfo:
    number_missing: int
    percentage_missing: float
    imputation_method: str

    def items(self):
        return asdict(self).items()


@dataclass
class MissingInfoMapping:
    missinginfo: dict[str, MissingInfo]

    # Convenience dict-like behavior
    def __getitem__(self, item: str) -> MissingInfo:
        return self.missinginfo[item]

    # Use a custom method to iterate over keys to avoid BaseModel __iter__ override issues
    def iter_keys(self):
        return iter(self.missinginfo)

    def items(self):
        return self.missinginfo.items()

    def keys(self):
        return self.missinginfo.keys()

    def values(self):
        return self.missinginfo.values()


@dataclass 
class MissingOverview:
  missinginfomapping: MissingInfoMapping
  missingplot: str
  missing_general: dict[str,int|float]


def suggest_imputation_methods(
    X_train: pd.DataFrame, columninfomapping: ColumnInfoMapping
) -> dict[str, list[str]]:
    """
    Suggests imputation methods for each feature in a DataFrame based on a set of rules.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict[str, list[str]]: A dictionary where keys are feature names and
                                values are lists of suggested imputation methods.
    """
    suggestions = {}
    num_features = X_train.shape[1]
    is_large_dataset = num_features > 100 or len(X_train) > 100_000

    # Pre-calculate correlation for numeric features
    numeric_df = X_train.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr().abs()

    for column in X_train.columns:
        feature = X_train[column]
        missing_percentage = (feature.isnull().sum() / len(feature)) * 100
        feature_type = columninfomapping[column].proposed_type

        if missing_percentage == 0:
            suggestions[column] = ["no imputation needed"]
            continue

        if missing_percentage > 70:
            suggestions[column] = ["drop", "flag for special handling"]
            continue

        # --- Numeric Feature Rules ---
        if feature_type == "numeric":
            is_highly_correlated = False
            if column in correlation_matrix:
                # Check if the feature is highly correlated with any other feature
                is_highly_correlated = (
                    correlation_matrix[column] > 0.7
                ).sum() > 1

            if missing_percentage <= 10:
                type_info = columninfomapping[column].type_specific_info
                if isinstance(type_info, NumericInfo) and hasattr(type_info, "skewness") and type_info.skewness > 1:
                    suggestions[column] = ["median"]
                else:
                    suggestions[column] = ["mean", "median"]
            elif 10 < missing_percentage <= 40:
                methods = ["kNN", "MissForest", "MICE"]
                if is_large_dataset:
                    methods.remove("MICE")
                if is_highly_correlated:
                    # Prioritize predictive methods
                    suggestions[column] = ["MICE", "MissForest"]
                else:
                    suggestions[column] = methods
            else:  # 40 < missing_percentage <= 70
                methods = ["MissForest", "MICE", "SoftImpute"]
                if is_large_dataset:
                    methods.remove("MICE")
                suggestions[column] = methods

        # --- Categorical Feature Rules ---
        else:  # feature_type == 'categorical':
            cardinality = feature.nunique()

            if missing_percentage <= 10:
                suggestions[column] = ["mode"]
            elif 10 < missing_percentage <= 40:
                if cardinality < 10:
                    suggestions[column] = ["mode", "kNN", "MissForest"]
                else:  # High cardinality
                    suggestions[column] = ["MissForest", "category embedding"]
            else:  # 40 < missing_percentage <= 70
                if cardinality < 10:
                    suggestions[column] = ["MissForest", "MICE"]
                else:  # High cardinality
                    suggestions[column] = ["MissForest", "category embedding"]

    return suggestions


def get_missing_info(
    original_data: OriginalData, columninfomapping: ColumnInfoMapping
) -> MissingOverview:
    """
    Generates a summary of missing values for each feature in a DataFrame.
    """
    df = original_data.X_train
    imputation_suggestions = suggest_imputation_methods(X_train=df, columninfomapping=columninfomapping)
    missinginfomapping={}
    for feature in df.columns:
        missing_count = df[feature].isna().sum()
        missing_percentage = df[feature].isna().mean() * 100
        missinginfo: MissingInfo = MissingInfo(
            number_missing=missing_count,
            percentage_missing=missing_percentage,
            imputation_method="<br>".join(imputation_suggestions[feature]),
        )
        missinginfomapping[feature] = missinginfo
    missing_count = df.isna().sum().sum()  # total missing values
    missing_percentage = (missing_count / df.size) * 100
    result = MissingOverview(
        missinginfomapping=MissingInfoMapping(missinginfo=missinginfomapping),
        missing_general={
            "Total number missing:": missing_count,
            "Total percentage missing": missing_percentage,
        },
        missingplot=plot_missingness_matrix(df),
    )
    return result
