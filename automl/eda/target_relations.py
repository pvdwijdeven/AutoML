# Standard library imports
from dataclasses import asdict, dataclass
from typing import Union, Any

# Third-party imports
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import stats
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
    target_type: str
    n_categories: int
    statistical_test: str
    effect_size: str
    comment: str
    
    def items(self):
        return asdict(self).items()


@dataclass
class CatCatCorr:
    statistical_test:str
    effect_size: str
    comment: str
    
    def items(self):
        return asdict(self).items()


@dataclass
class TargetRelationMapping:
    target_relation: dict[
        str, Union[NumNumCorr, CatNumCorr, NumCatCorr, CatCatCorr]
    ]


def generate_interpretation(
    analysis_dict: dict[str, Any], feature_name: str, target_name: str
) -> str:
    """
    Generates a human-readable interpretation from an analysis dictionary.

    Args:
        analysis_dict (Dict[str, Any]): The output dictionary from one of the
                                        analysis functions.
        feature_name (str): The name of the feature variable.
        target_name (str): The name of the target variable.

    Returns:
        str: A text summary of the statistical findings.
    """
    if "error" in analysis_dict:
        return f"Analysis could not be performed for '{feature_name}' and '{target_name}': {analysis_dict['error']}"

    # --- Step 1: Check for Statistical Significance ---
    p_value = analysis_dict["statistical_test"]["p_value"]
    test_name = analysis_dict["statistical_test"]["name"]
    alpha = 0.05

    # Format p-value for reporting
    p_text = f"p = {p_value:.4f}" if p_value >= 0.001 else "p < 0.001"

    if p_value >= alpha:
        return (
            f"The {test_name} found no statistically significant relationship between "
            f"'{feature_name}' and '{target_name}' ({p_text}). Any observed "
            f"differences or associations are likely due to random chance."
        )

    # --- Step 2: If significant, interpret the effect size ---
    effect_size_name = analysis_dict["effect_size"]["name"]
    effect_size_value = analysis_dict["effect_size"]["value"]
    strength = "unknown"

    if effect_size_name == "Cramér's V":
        if effect_size_value >= 0.5:
            strength = "strong"
        elif effect_size_value >= 0.3:
            strength = "moderate"
        elif effect_size_value >= 0.1:
            strength = "weak"
        else:
            strength = "very weak"

        return (
            f"A statistically significant and <strong>{strength} association</strong> was found "
            f"between '{feature_name}' and '{target_name}' ({p_text}, "
            f"Cramér's V = {effect_size_value:.2f})."
        )

    elif effect_size_name == "Eta-squared (η²)":
        if effect_size_value >= 0.14:
            strength = "large"
        elif effect_size_value >= 0.06:
            strength = "moderate"
        elif effect_size_value >= 0.01:
            strength = "small"
        else:
            strength = "very small"

        return (
            f"A statistically significant relationship was found. The groups in "
            f"'{target_name}' account for a <strong>{strength}</strong> amount of the variance in "
            f"'{feature_name}' ({p_text}, η² = {effect_size_value:.2f})."
        )

    elif effect_size_name == "Cohen's d":
        abs_d = abs(effect_size_value)
        if abs_d >= 0.8:
            strength = "large"
        elif abs_d >= 0.5:
            strength = "moderate"
        elif abs_d >= 0.2:
            strength = "small"
        else:
            strength = "very small"

        return (
            f"A statistically significant and <strong>{strength} difference</strong> was found in the "
            f"mean of '{feature_name}' between the categories of '{target_name}' "
            f"({p_text}, Cohen's d = {effect_size_value:.2f})."
        )

    return "Interpretation could not be generated."


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


def get_cat_cat_correlation(
    feature: pd.Series, target: pd.Series
) -> CatCatCorr:
    """
    Analyzes the relationship between two categorical variables.

    This function creates a contingency table, performs a Chi-Squared test for
    independence, and calculates Cramér's V as a measure of the strength of
    the association.

    Args:
        categorical_feature (pd.Series): A pandas Series containing the feature's
                                            categorical data.
        categorical_target (pd.Series): A pandas Series containing the target's
                                        categorical data. Must be of the same
                                        length as categorical_feature.

    Returns:
        Dict[str, Any]: A dictionary containing the contingency table, Chi-Squared
                        test results, and the Cramér's V effect size. Returns an
                        error message if analysis is not possible.
    """
    if len(feature) != len(target):
        raise ValueError("Input Series must have the same length.")

    # Combine into a single DataFrame and drop missing values
    df = pd.DataFrame(
        {"feature": feature, "target": target}
    ).dropna()
    skip = False
    comment = ""
    if df.empty:
        comment = "No valid data after dropping missing values."
        skip = True

    # --- 1. Contingency Table ---
    # This table shows the frequency of each combination of categories.
    contingency_table = pd.crosstab(df["feature"], df["target"])

    # Check if the table is valid for a chi-squared test
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        comment = "At least one variable has fewer than 2 unique categories after dropping NaNs."
        skip = True

    if not skip:
        # --- 2. Statistical Test: Chi-Squared ---
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # --- 3. Effect Size: Cramér's V ---
        # Cramér's V is a measure of association between two nominal variables,
        # giving a value between 0 (no association) and 1 (perfect association).
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape

        # Handle the case where a dimension has only 1 level
        if min(k - 1, r - 1) == 0:
            cramers_v = np.nan
        else:
            cramers_v = np.sqrt(phi2 / min(k - 1, r - 1))

        # --- 4. Assemble Results ---
        results = {
            "statistical_test": {
                "name": "Chi-Squared Test of Independence",
                "chi2_statistic": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
            },
            "effect_size": {"name": "Cramér's V", "value": cramers_v},
        }
        feature_name = str(feature.name) if feature.name else "feature"
        target_name = str(target.name) if target.name else "target"

        comment = generate_interpretation(
            analysis_dict=results,
            feature_name=feature_name,
            target_name=target_name,
        )
        result = CatCatCorr(
            statistical_test=(
                f"name: Chi-Squared Test of Independence<br>chi2_statistic: {chi2}<br>p_value: {p_value}<br>degrees_of_freedom: {dof}"
            ),
            effect_size=f"name: Cramér's V<br>value: {cramers_v}",
            comment=comment,
        )
    else:
        result = CatCatCorr(statistical_test="", effect_size="",comment=comment)
    return result


def get_num_cat_correlation(
    feature: pd.Series, target: pd.Series
) -> NumCatCorr:
    """
    Analyzes the relationship between a numeric feature and a categorical target.

    This function provides descriptive statistics for the numeric feature grouped by each
    category of the target. It automatically performs the appropriate statistical test
    (t-test for binary target, ANOVA for multi-class target) and calculates a
    corresponding effect size (Cohen's d or Eta-squared).

    Args:
        numeric_feature (pd.Series): A pandas Series containing numerical data.
        categorical_target (pd.Series): A pandas Series containing categorical data.
                                        Must be of the same length as numeric_feature.

    Returns:
        Dict[str, Any]: A dictionary containing a comprehensive analysis of the
                        relationship, including summary stats, test results, and
                        effect size. Returns an error message if analysis
                        is not possible.
    """
    if len(feature) != len(target):
        raise ValueError("Input Series must have the same length.")

    # Combine into a single DataFrame and drop missing values
    df = pd.DataFrame(
        {"feature": feature, "target": target}
    ).dropna()

    skip = False
    comment = ""

    if df.empty:
        comment = "No valid data after dropping missing values."
        skip = True

    # Identify categories and their count
    categories = df["target"].unique()
    n_categories = len(categories)

    if n_categories < 2:
        comment = f"Target must have at least 2 unique categories, but found {n_categories}."
        skip = True

    # --- 1. Summary Statistics by Category ---

    results = NumCatCorr(        target_type= "binary" if n_categories == 2 else "multiclass",
        n_categories= n_categories,
        statistical_test= "",
        effect_size= "",
        comment=comment,
    )
    my_dict = {
        "name": "",
        "statistical_test": {},
        "effect_size": {}
    }
    if not skip:
        # --- 2. Statistical Test and Effect Size ---
        if n_categories == 2:
            # For Binary Targets
            group1 = df[df["target"] == categories[0]]["feature"]
            group2 = df[df["target"] == categories[1]]["feature"]

            # Welch's T-test (does not assume equal variance)
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            results.statistical_test = f"name: Welch's Independent T-test<br>t_statistic: {t_stat}<br>p_value: {p_value}"
            my_dict["statistical_test"] = {
                "name": "Welch's Independent T-test",
                "t_statistic": t_stat,
                "p_value": p_value,
            }
            # Cohen's d for effect size
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = group1.mean(), group2.mean()
            std1, std2 = group1.std(), group2.std()
            pooled_std = np.sqrt(
                ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
            )
            cohen_d = (mean1 - mean2) / pooled_std
            results.effect_size = f"name: Cohen's d<br>value: {cohen_d}"
            my_dict["effect_size"] = {
            "name": "Cohen's d",
            "value": cohen_d
        }
        else:  # n_categories > 2
            # For Multi-class Targets
            groups = [df[df["target"] == cat]["feature"] for cat in categories]

            # ANOVA Test
            f_stat, p_value = stats.f_oneway(*groups)
            results.statistical_test = f"name: One-Way ANOVA<br>F_statistic: {f_stat}<br>p_value: {p_value}"
            my_dict["statistical_test"] = {
                "name": "One-Way ANOVA",
                "F_statistic": f_stat,
                "p_value": p_value,
            }
            # Eta-squared for effect size
            ss_between = sum(
                len(g) * (g.mean() - df["feature"].mean()) ** 2 for g in groups
            )
            ss_total = sum((x - df["feature"].mean()) ** 2 for x in df["feature"])
            eta_squared = ss_between / ss_total
            results.effect_size = f"name: Eta-squared (η²)<br>value: {eta_squared}"
            my_dict["effect_size"] = {
                "name": "Eta-squared (η²)",
                "value": eta_squared,
            }
        feature_name = str(feature.name) if feature.name else "feature"
        target_name = str(target.name) if target.name else "target"

        results.comment = generate_interpretation(
            analysis_dict=my_dict,
            feature_name=feature_name,
            target_name=target_name,
        )

    return results


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
