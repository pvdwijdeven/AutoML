# Standard library imports
from dataclasses import dataclass, asdict
from typing import Union

# Third-party imports
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import pearsonr, shapiro, spearmanr

from .column_analysis import ColumnInfoMapping


@dataclass
class NumNumCorr:
    pearson_correlation: float
    pearson_pvalue: float
    spearman_correlation: float
    spearman_pvalue: float
    recommended_correlation: str
    reason: str
    interpretation: str

    def items(self):
        return asdict(self).items()


@dataclass
class CatNumCorr:
    correlation_metric_name: str
    correlation_metric_value: float
    mean_target_per_category: str
    category_counts: str
    insights_and_recommendations: str

    def items(self):
        return asdict(self).items()


@dataclass
class NumCatCorr:

    def items(self):
        return asdict(self).items()


@dataclass
class CatCatCorr:

    def items(self):
        return asdict(self).items()


@dataclass
class TargetRelationMapping:
    target_relation: dict[
        str, Union[NumNumCorr, CatNumCorr, NumCatCorr, CatCatCorr]
    ]


# This is the helper function you provided, used for generating the final interpretation.
def interpret_correlation(
    coef: float, pvalue: float, method: str = "pearson"
) -> str:
    """
    Generate a human-readable interpretation of a correlation coefficient and p-value.
    """
    # Choose method label
    method = method.lower()
    if method == "pearson":
        method_label = "linear correlation (Pearson r)"
    elif method == "spearman":
        method_label = "monotonic correlation (Spearman ρ)"
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    # Strength interpretation
    abs_coef = abs(coef)
    if abs_coef < 0.20:
        strength = "very weak"
    elif abs_coef < 0.40:
        strength = "weak"
    elif abs_coef < 0.60:
        strength = "moderate"
    elif abs_coef < 0.80:
        strength = "strong"
    else:
        strength = "very strong"

    # Direction
    if coef > 0:
        direction = "positive"
    elif coef < 0:
        direction = "negative"
    else:
        direction = "no"

    # Significance interpretation
    if pvalue > 0.05:
        significance = f"not statistically significant (p = {pvalue:.3f})"
    elif pvalue > 0.01:
        significance = f"statistically significant (p = {pvalue:.3f})"
    elif pvalue > 0.001:
        significance = f"strongly significant (p = {pvalue:.3f})"
    else:
        significance = "very strongly significant (p < 0.001)"

    if np.isnan(coef):
        return "Not enough data points to calculate correlation."

    # Build interpretation
    return (
        f"{strength.capitalize()} {direction} {method_label}: "
        f"{coef:.2f}, {significance}."
    )


def _get_eta_squared(x: pd.Series, y: pd.Series) -> float:
    """
    Calculates the Correlation Ratio (Eta-squared) for a categorical feature
    and a numerical target.

    Args:
        x (pd.Series): The categorical feature (or predictor).
        y (pd.Series): The numerical target (or response).

    Returns:
        float: The Eta-squared value, ranging from 0 to 1.
    """
    # Drop missing values pair-wise
    data = pd.concat([x, y], axis=1).dropna()
    x_clean = data.iloc[:, 0]
    y_clean = data.iloc[:, 1]

    if y_clean.nunique() <= 1:
        return 0.0  # No variance in target, so no explainable variance.

    total_mean = y_clean.mean()
    ss_total = ((y_clean - total_mean) ** 2).sum()

    if ss_total == 0:
        return 0.0  # Avoid division by zero.

    groups = y_clean.groupby(x_clean)
    ss_between = sum(
        len(group) * ((group.mean() - total_mean) ** 2) for _, group in groups
    )

    return ss_between / ss_total


def get_cat_num_correlation(
    feature: pd.Series,
    target: pd.Series,
) -> CatNumCorr:
    """
    Analyzes the relationship between a categorical feature and a numerical target,
    returning a dictionary with statistics and textual insights.

    Args:
        x (pd.Series): The categorical feature.
        y (pd.Series): The numerical target.
        feature_name (str): The name of the feature for use in textual insights.
        target_name (str): The name of the target for use in textual insights.

    Returns:
        Dict[str, Any]: A dictionary containing relational metrics and insights.
    """
    feature_name: str = str(object=feature.name)
    target_name: str = str(object=target.name)
    # --- 1. Statistical Calculations ---
    data = pd.concat([feature, target], axis=1).dropna()
    x_clean = data.iloc[:, 0]
    y_clean = data.iloc[:, 1]

    if x_clean.empty or y_clean.empty:
        return CatNumCorr(
        correlation_metric_name="",
        correlation_metric_value=0,
        mean_target_per_category="",
        category_counts="",
        insights_and_recommendations="Too few parameters after removing missing values",
    )

    # Calculate correlation ratio for ranking
    eta_squared = _get_eta_squared(x_clean, y_clean)

    # Group target by feature categories
    grouped_target = y_clean.groupby(x_clean)
    mean_target_per_category = grouped_target.mean().to_dict()
    category_counts = grouped_target.count().to_dict()

    # --- 2. Generate Textual Insights ---
    insights = []

    # Classify strength of relationship
    if eta_squared >= 0.14:
        strength = "strong"
    elif eta_squared >= 0.06:
        strength = "moderate"
    elif eta_squared >= 0.01:
        strength = "weak"
    else:
        strength = "very weak or no"

    insights.append(
        f"The feature '{feature_name}' has a {strength} relationship with "
        f"the target '{target_name}' (η² = {eta_squared:.3f})."
    )

    # Find highest and lowest impact categories
    if mean_target_per_category:
        highest_cat = max(
            mean_target_per_category, key=lambda k: mean_target_per_category[k]
        )
        lowest_cat = min(
            mean_target_per_category, key=lambda k: mean_target_per_category[k]
        )
        insights.append(
            f"The highest average target value is observed in the '{highest_cat}' category "
            f"({mean_target_per_category[highest_cat]:.2f}), while the lowest is in the "
            f"'{lowest_cat}' category ({mean_target_per_category[lowest_cat]:.2f})."
        )

    # Check for small categories that might be unreliable
    total_samples = sum(category_counts.values())
    small_categories = [
        cat
        for cat, count in category_counts.items()
        if count
        < max(
            10, total_samples * 0.05
        )  # A category is small if < 10 samples or < 5% of total
    ]
    if small_categories:
        insights.append(
            f"Note: The following categories have a small sample size, which may affect "
            f"the reliability of their statistics: {', '.join(map(str, small_categories))}."
        )

    # --- 3. Assemble Final Dictionary ---

    return CatNumCorr(
        correlation_metric_name="eta_squared",
        correlation_metric_value=eta_squared,
        mean_target_per_category= "<br>".join(f"{k}: {v}" for k, v in mean_target_per_category.items()),
        category_counts="<br>".join(f"{k}: {v}" for k, v in category_counts.items()),
        insights_and_recommendations="<br>".join(insights),
    )


def get_num_cat_correlation(
    feature: pd.Series,
    target: pd.Series,
) -> NumCatCorr:
    result = NumCatCorr()
    return result


def get_cat_cat_correlation(
    feature: pd.Series,
    target: pd.Series,
) -> CatCatCorr:
    result = CatCatCorr()
    return result


def get_num_num_correlation(
    feature: pd.Series, target: pd.Series, alpha: float = 0.05
) -> NumNumCorr:
    """
    Selects the most appropriate correlation coefficient (Pearson or Spearman)
    based on the data's characteristics and returns a comprehensive analysis.

    Parameters:
    - feature (pd.Series): The independent variable series (assumed numeric).
    - target (pd.Series): The dependent variable series (assumed numeric).
    - alpha (float): Significance level for statistical tests.

    Returns:
    - NumNumCorr: A dataclass containing both correlation results, the recommendation,
            the reasoning, and a human-readable interpretation.
    """
    # --- Step 1: Data Cleaning and Size Check ---
    combined_df = pd.DataFrame({"feature": feature, "target": target}).dropna()
    feature_clean = combined_df["feature"]
    target_clean = combined_df["target"]

    # If too few data points after removing NaNs, return a dictionary with empty values.
    if len(feature_clean) < 3:
        return NumNumCorr(
            pearson_correlation=np.nan,
            pearson_pvalue=np.nan,
            spearman_correlation=np.nan,
            spearman_pvalue=np.nan,
            recommended_correlation="None",
            reason="Too few feature data for correlation values",
            interpretation="",
        )

    # --- Step 2: Calculate Both Correlation Coefficients ---
    # Always calculate both to include in the final output.
    r_p, p_p = pearsonr(feature_clean, target_clean)
    r_s, p_s = spearmanr(feature_clean, target_clean)
    assert isinstance(r_p, float)
    assert isinstance(r_s, float)
    assert isinstance(p_p, float)
    assert isinstance(p_s, float)
    # --- Step 3: Statistical Tests for Assumptions ---

    # Shapiro-Wilk test for normality
    feature_is_normal = shapiro(feature_clean).pvalue > alpha
    target_is_normal = shapiro(target_clean).pvalue > alpha

    # Check for outliers using the IQR method
    def has_outliers(s: pd.Series) -> bool:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return bool(((s < lower_bound) | (s > upper_bound)).any())

    feature_has_outliers = has_outliers(feature_clean)
    target_has_outliers = has_outliers(target_clean)

    # --- Step 4: Core Decision Logic ---
    reason = ""
    recommended_method = ""

    if not (feature_is_normal and target_is_normal):
        recommended_method = "Spearman"
        reason = (
            "One or both variables fail the Shapiro-Wilk normality test. "
            "Spearman's correlation is a robust, non-parametric measure that "
            "does not assume normality."
        )
    elif feature_has_outliers or target_has_outliers:
        recommended_method = "Spearman"
        reason = (
            "Outliers were detected in at least one variable. Pearson's correlation is "
            "highly sensitive to outliers, whereas Spearman's rank correlation is robust "
            "to them, providing a more reliable measure."
        )

    elif abs(r_p - r_s) > 0.2:
        recommended_method = "Spearman"
        reason = (
            f"The data is normal and free of outliers, but the difference between "
            f"Pearson ({r_p:.3f}) and Spearman ({r_s:.3f}) is substantial. "
            "This suggests a non-linear but monotonic relationship, which Spearman "
            "captures more effectively."
        )
    else:
        recommended_method = "Pearson"
        reason = (
            "All assumptions for Pearson's correlation are met (normality, no outliers). "
            f"The Pearson ({r_p:.3f}) and Spearman ({r_s:.3f}) coefficients are similar, "
            "confirming a linear relationship. Pearson is the most powerful test in this case."
        )

    # --- Step 5: Generate Interpretation and Final Output ---
    if recommended_method == "Pearson":
        final_coef, final_pvalue = r_p, p_p
    else:
        final_coef, final_pvalue = r_s, p_s

    interpretation_text = interpret_correlation(
        coef=final_coef, pvalue=final_pvalue, method=recommended_method
    )

    return NumNumCorr(
        pearson_correlation=r_p,
        pearson_pvalue=p_p,
        spearman_correlation=r_s,
        spearman_pvalue=p_s,
        recommended_correlation=recommended_method,
        reason=reason,
        interpretation=interpretation_text,
    )


def get_target_relations(
    X_train: DataFrame, y_train: Series, column_info: ColumnInfoMapping
) -> TargetRelationMapping:

    target_relation = {}
    target_type = column_info["target"].proposed_type
    for column in X_train.columns:
        feature_type = column_info[column].proposed_type
        # todo extend with other types
        if target_type == "numeric":
            if feature_type == "numeric":
                target_relation[column] = get_num_num_correlation(
                    feature=X_train[column], target=y_train
                )
            if feature_type == "categorical":
                target_relation[column] = get_cat_num_correlation(
                    feature=X_train[column], target=y_train
                )
        if target_type == "categorical":
            if feature_type == "numeric":
                target_relation[column] = get_num_cat_correlation(
                    feature=X_train[column], target=y_train
                )
            if feature_type == "categorical":
                target_relation[column] = get_cat_cat_correlation(
                    feature=X_train[column], target=y_train
                )
    return TargetRelationMapping(**{"target_relation": target_relation})
