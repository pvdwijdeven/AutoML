# Standard library imports
import warnings
from dataclasses import dataclass
from typing import Optional

# Third-party imports
import numpy as np
from pandas import DataFrame
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# Local application imports
from automl.dataloader import OriginalData
from automl.library import Logger


@dataclass
class RelationInfo():
    mutual_information: str
    related_features: str
    suggestions: str


@dataclass
class RelationMapping():
    relations: dict[str, RelationInfo] 
    number_of_features: int = 0



def select_features_by_missingness(
    X_work: DataFrame,
    candidate_features: list[str] = [],
    max_features: int = 100,
    min_row_fraction: float = 0.2,
) -> list[str]:
    """
    Select features sorted by ascending missing values, adding features
    until max_features is reached or dropping NA rows keeps at least min_row_fraction of data.

    Parameters:
    - df: DataFrame with your data
    - candidate_features: list of feature column names to consider
    - max_features: max number of features to select
    - min_row_fraction: minimal fraction of rows to keep after dropping NA

    Returns:
    - selected_features: list of selected features satisfying conditions
    """
    if not candidate_features:
        # If no specific features provided, use all columns
        candidate_features = X_work.columns.tolist()
    # Sort features by missing values ascending
    missing_counts = X_work[candidate_features].isna().sum().sort_values()

    selected_features = []
    original_row_count = len(X_work)

    for feature in missing_counts.index:
        selected_features.append(feature)

        # Calculate how many rows remain after dropping NA in current features
        rows_remaining = X_work.dropna(subset=selected_features).shape[0]
        fraction_remaining = rows_remaining / original_row_count

        # Check stop conditions
        if (
            len(selected_features) >= max_features
            or fraction_remaining <= min_row_fraction
        ):
            # If fraction dropped below threshold after adding last feature,
            # remove that feature to keep fraction above min_row_fraction
            if fraction_remaining < min_row_fraction:
                selected_features.pop()
            break

    return selected_features


def generate_feature_relations(
    original_data: OriginalData,
    max_features: int = 100,
    max_samples: int = 10000,
    logger: Optional[Logger] = None,
) -> RelationMapping:

    warnings.filterwarnings("ignore")

    X_work = original_data.X_train.copy()

    target_name = str(object=original_data.y_train.name)
    X_work[target_name] = original_data.y_train.copy()

    # Step 1: Select features and target
    features = select_features_by_missingness(X_work=X_work)
    if target_name not in features:
        features.append(target_name)

    # Step 2: Drop missing values only for selected features
    X_work = X_work[features].dropna()

    # Step 3: Sample for performance
    if len(X_work) > max_samples:
        X_work = X_work.sample(n=max_samples, random_state=42)

    # Step 4: Identify numeric and categorical columns
    num_cols = X_work.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_work.select_dtypes(exclude=np.number).columns.tolist()

    # Step 5: Label encode categorical features
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_work[col] = X_work[col].astype(str)
        X_work[col] = le.fit_transform(X_work[col])
        le_dict[col] = le

    # Step 6: Restrict to max_features
    all_features = sorted(num_cols + cat_cols, key=str.lower)

    if len(all_features) > max_features:
        all_features = all_features[:max_features]

    # Step 7: Correlation
    corr_matrix = X_work[all_features].corr()

    # Step 8: Mutual Information
    if logger:
        logger.info("[GREEN]  - mutual information.")
    mi_scores = {}
    try:
        X_features = [f for f in all_features if f != target_name]
        if target_name and target_name in X_work.columns:
            y = X_work[target_name]
            X = X_work[X_features]
            if y.nunique() <= 10:
                mi = mutual_info_classif(
                    X, y, discrete_features="auto", random_state=42
                )
            else:
                mi = mutual_info_regression(
                    X, y, discrete_features="auto", random_state=42
                )
            mi_scores = dict(zip(X_features, mi))
    except Exception as e:
        print(f"Error computing mutual information: {e}")
        mi_scores = {feature: None for feature in all_features}

    # Step 9: Aggregate results
    if logger:
        logger.info("[GREEN]- Creating relation info per feature.")
    insights = {}
    for feature in all_features:
        related = []
        suggestions = []

        # Related by correlation
        if feature in corr_matrix:
            high_corr = corr_matrix[feature][
                (corr_matrix[feature].abs() > 0.7)
                & (corr_matrix[feature].abs() < 1.0)
            ]
            related = high_corr.index.tolist()
            if not high_corr.empty:
                suggestions.append(
                    "Highly correlated with "
                    + ", ".join(related)
                    + " — consider dropping or combining."
                )

        # Mutual information
        mi_score = mi_scores.get(feature)
        if mi_score is not None:
            if mi_score > 0.2:
                suggestions.append(
                    f"High mutual information with target (MI={mi_score:.2f}) — useful feature."
                )
            elif mi_score < 0.01:
                suggestions.append(
                    f"Low mutual information with target (MI={mi_score:.2f}) — may not help predictive power."
                )

        insights[feature] = {
            "related_features": "<br>".join(related) if related else "None",
            "mutual_information": (
                f"{mi_score:.3f}" if mi_score is not None else "N/A"
            ),
            "suggestions": (
                "- " + "<br>- ".join(suggestions) if suggestions else "None"
            ),
        }

    return RelationMapping(
        **{"relations": insights, "number_of_features": len(all_features)}
    )
