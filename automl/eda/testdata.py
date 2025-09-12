# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np
from pandas import DataFrame, Series

# Local application imports
from automl.dataloader import OriginalData


@dataclass
class TestInfo():
    df_size: DataFrame
    df_colcheck: DataFrame
    df_missing: DataFrame
    df_unseen_cats: DataFrame
    df_num_stats: DataFrame
    df_type_issues: DataFrame
    suggestions: dict[str, str]


def analyze_test_data(original_data: OriginalData) -> TestInfo:

    def suggestion(row):
        test_missing = (
            float(row["Missing % (Test)"])
            if row["Missing % (Test)"] != "-"
            else 0.0
        )
        train_missing = (
            float(row["Missing % (Train)"])
            if row["Missing % (Train)"] != "-"
            else 0.0
        )

        if test_missing == 0.0:
            return ""
        elif test_missing > 0.0 and train_missing > 0.0:
            return "Use training missing handling method"
        elif test_missing > 0.0 and train_missing == 0.0:
            return "Use imputation for missing test data: no imputation in training data required."
        else:
            return ""  # Optional, for completeness

    suggestions = {}
    assert original_data.X_comp is not None
    X_train = original_data.X_train.copy()
    X_comp = original_data.X_comp.copy()
    # 1. Size comparison
    rows_train, cols_train = X_train.shape
    rows_test, cols_test = X_comp.shape
    df_size = DataFrame(
        data={
            "Dataset": ["Train", "Test"],
            "Rows": [rows_train, rows_test],
            "Columns": [cols_train, cols_test],
        }
    )

    if rows_test < 0.5 * rows_train:
        suggestions["size"] = (
            "The test set is much smaller than the training set (< 50%). Results might be unstable or under-represented."
        )
    elif rows_test > 2 * rows_train:
        suggestions["size"] = (
            "The test set is much larger (>100%) than the training set. Consider whether the training data is representative enough."
        )

    # 2. Column comparison
    train_cols = set(X_train.columns)
    test_cols = set(X_comp.columns)
    only_in_train = sorted(train_cols - test_cols)
    only_in_test = sorted(test_cols - train_cols)
    common_cols = sorted(train_cols & test_cols)
    df_colcheck = DataFrame(
        {
            "Only in Train": Series(only_in_train),
            "Only in Test": Series(only_in_test) if only_in_test else "None",
        }
    )
    if len(only_in_train) > 0:
        suggestions["colcheck"] = (
            "Ensure column alignment. Consider dropping training columns."
        )
    if len(only_in_test) > 0:
        suggestions["colcheck"] = (
            "Ensure column alignment. Consider dropping test columns or filling train columns."
        )

    # 3. Missing values
    missing_train = X_train.isna().mean().round(3) * 100
    missing_test = X_comp.isna().mean().round(3) * 100
    df_missing = DataFrame(
        {
            "Missing % (Train)": missing_train,
            "Missing % (Test)": missing_test,
        }
    ).fillna("-")
    df_missing["suggestion"] = df_missing.apply(suggestion, axis=1)

    df_missing = df_missing[df_missing["Missing % (Test)"] > 0].sort_values(
        by="Missing % (Test)", ascending=False
    )

    # 4. Categorical: unseen categories
    num_cols = sorted(
        X_train.select_dtypes(include=np.number).columns, key=str.lower
    )
    num_cols = sorted(set(num_cols) & set(common_cols), key=str.lower)
    cat_cols = sorted(set(common_cols) - set(num_cols), key=str.lower)

    unseen_data = []
    for col in cat_cols:
        if col in X_comp.columns:
            train_cats = set(X_train[col].dropna().astype(str).unique())
            test_cats = set(X_comp[col].dropna().astype(str).unique())
            unseen = test_cats - train_cats
            if unseen:
                unseen_data.append(
                    {
                        "Column": col,
                        "Unseen Categories in Test": ", ".join(sorted(unseen))[
                            :300
                        ]
                        + ("..." if len(", ".join(unseen)) > 300 else ""),
                    }
                )
    df_unseen_cats = DataFrame(unseen_data)
    if unseen_data:
        suggestions["unseen_cats"] = (
            "The model might fail or misinterpret unseen categories. Use encoders that support unknowns (e.g., `handle_unknown='ignore'` in sklearn’s OneHotEncoder or fallback labels in target encoding)."
        )

    # 5. Numeric distributions (with mean/std/min/max shift detection and inline coloring)
    num_cols = sorted(
        X_train.select_dtypes(include=np.number).columns, key=str.lower
    )
    stats = []
    for col in num_cols:
        if col in X_comp.columns:
            train_mean = X_train[col].mean()
            test_mean = X_comp[col].mean()
            train_std = X_train[col].std()
            test_std = X_comp[col].std()
            train_min = X_train[col].min()
            test_min = X_comp[col].min()
            train_max = X_train[col].max()
            test_max = X_comp[col].max()

            train_range = train_max - train_min
            range_margin = 0.2 * train_range if train_range > 0 else 0

            min_shift_flag = abs(test_min - train_min) > range_margin
            max_shift_flag = abs(test_max - train_max) > range_margin

            mean_shift_flag = (
                abs(train_mean - test_mean) > 0.2 * train_std
                if train_std > 0
                else False
            )
            std_shift_flag = (
                abs(train_std - test_std) > 0.2 * train_std
                if train_std > 0
                else False
            )

            stats.append(
                {
                    "Column": col,
                    "Train Mean": round(train_mean, 3),
                    "Test Mean": round(test_mean, 3),
                    "Train Std": round(train_std, 3),
                    "Test Std": round(test_std, 3),
                    "Train Min": round(train_min, 3),
                    "Test Min": round(test_min, 3),
                    "Train Max": round(train_max, 3),
                    "Test Max": round(test_max, 3),
                    "Mean Shift": mean_shift_flag,
                    "Std Shift": std_shift_flag,
                    "Min Shift": min_shift_flag,
                    "Max Shift": max_shift_flag,
                }
            )

    df_num_stats = DataFrame(stats)

    suggestions["num_stats"] = (
        """Cells highlighted in light red indicate significant distribution shifts between training and test data.
        Cells highlighted in light red indicate significant distribution shifts between training and test data.
        Mean/Std: Could indicate changes in data patterns. Consider normalization or retraining.
        Values fall outside expected range. May cause extrapolation errors for some models (e.g., tree splits, linear assumptions)."""
    )

    # 6. Data type mismatches
    dtype_issues = []
    for col in X_train.columns.intersection(X_comp.columns):
        train_type = str(X_train[col].dtype)
        test_type = str(X_comp[col].dtype)
        if train_type != test_type:
            dtype_issues.append(
                {
                    "Column": col,
                    "Train Dtype": train_type,
                    "Test Dtype": test_type,
                }
            )
    df_type_issues = DataFrame(dtype_issues)
    if dtype_issues:
        suggestions["type_issues"]= (
            "Apply consistent dtype conversions. Data type mismatches may break transformation or prediction pipelines."
        )
    return TestInfo(
        **{
            "df_size": df_size,
            "df_colcheck": df_colcheck,
            "df_missing": df_missing,
            "df_unseen_cats": df_unseen_cats,
            "df_num_stats": df_num_stats,
            "df_type_issues": df_type_issues,
            "suggestions": suggestions,
        }
    )
