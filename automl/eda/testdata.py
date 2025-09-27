# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np
import pandas as pd 
from pandas import DataFrame, Series

# Local application imports
from automl.dataloader import OriginalData
from .column_analysis import ColumnInfoMapping


@dataclass
class TestInfo:
    df_size: DataFrame
    html_size: str
    df_colcheck: DataFrame
    html_colcheck: str
    df_missing: DataFrame
    html_missing: str
    df_unseen_cats: DataFrame
    html_unseen_cats: str
    df_num_stats: DataFrame
    html_num_stats: str
    df_type_issues: DataFrame
    html_type_issues: str
    suggestions: dict[str, str]
    samples_head: str
    samples_middle: str
    samples_tail: str


def build_shift_table(stats: list[dict]) -> str:
    """
    Build an HTML table from numeric shift stats with Bootstrap styling.
    Cells with shifts are highlighted using Bootstrap variables.
    """
    df = pd.DataFrame(stats)

    # Columns to display (skip the flag columns)
    display_cols = [
        "Column",
        "Train Mean",
        "Test Mean",
        "Train Std",
        "Test Std",
        "Train Min",
        "Test Min",
        "Train Max",
        "Test Max",
    ]

    html = ['<table class="frequency-table">']
    # Header
    html.append("<thead><tr>")
    for col in display_cols:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead>")

    # Body
    html.append("<tbody>")
    for _, row in df.iterrows():
        html.append("<tr>")
        for col in display_cols:
            style = ""
            # Highlight based on flag
            if col == "Test Mean" and row["Mean Shift"]:
                style = ' style="background-color: var(--bs-danger); color: var(--bs-light);"'
            elif col == "Test Std" and row["Std Shift"]:
                style = ' style="background-color: var(--bs-danger); color: var(--bs-light);"'
            elif col == "Test Min" and row["Min Shift"]:
                style = ' style="background-color: var(--bs-danger); color: var(--bs-light);"'
            elif col == "Test Max" and row["Max Shift"]:
                style = ' style="background-color: var(--bs-danger); color: var(--bs-light);"'
            html.append(f"<td{style}>{row[col]}</td>")
        html.append("</tr>")
    html.append("</tbody>")
    html.append("</table>")

    return "\n".join(html)


def analyze_test_data(original_data: OriginalData, column_info: ColumnInfoMapping) -> TestInfo:

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
    percent = round(rows_test / rows_train * 100, 1)
    df_size = DataFrame(
        data={
            "Dataset": ["Train", "Test"],
            "Rows": [rows_train, f"{rows_test} ({percent}% of Training set)"],
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
            "Only in Train": (
                Series(only_in_train) if only_in_train else Series("None")
            ),
            "Only in Test": (
                Series(only_in_test) if only_in_test else Series("None")
            ),
        }
    )
    suggestions["colcheck"] = ""
    if len(only_in_train) > 0:
        suggestions["colcheck"] = (
            "Ensure column alignment. Consider dropping training columns."
        )
    if len(only_in_test) > 0:
        suggestions["colcheck"] += (
            "Ensure column alignment. Consider dropping test columns or filling train columns."
        )

    # 3. Missing values
    missing_train = X_train[common_cols].isna().mean().round(3) * 100
    missing_test = X_comp[common_cols].isna().mean().round(3) * 100
    df_missing = DataFrame(
        {"Column": common_cols,
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
            if column_info[col].proposed_type == "categorical":
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
    else:
        suggestions["unseen_cats"] = "No issues regarding unseen categories."

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

    df_num_stats = pd.DataFrame(stats)

    table_html = build_shift_table(stats=stats)
    
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
                    "Remark": "Only float64 can be used in case of missing data" if test_type in ["Int64","float64"] and train_type in ["Int64","float64"] else "Check contents"
                }
            )
    df_type_issues = DataFrame(dtype_issues)
    if dtype_issues:
        suggestions["type_issues"] = (
            "Apply consistent dtype conversions. Data type mismatches may break transformation or prediction pipelines."
        )
    else:
        suggestions["type_issues"] = "No type issues found."
    n_rows = len(X_comp)
    samples_head = (
        X_comp.head(10)
        .to_html(index=False, na_rep="<N/A>")
        .replace('border="1"', "")
    )
    samples_middle = (
        X_comp.iloc[n_rows // 2 - 5 : n_rows // 2 + 5]
        .to_html(index=False, na_rep="<N/A>")
        .replace('border="1"', "")
    )
    samples_tail = (
        X_comp.tail(10)
        .to_html(index=False, na_rep="<N/A>")
        .replace('border="1"', "")
    )
    return TestInfo(
        **{
            "df_size": df_size,
            "html_size": df_size.to_html(
                index=False, classes=["table-wrapper2"]
            ),
            "df_colcheck": df_colcheck,
            "html_colcheck": df_colcheck.to_html(
                index=False, classes=["table-wrapper2"]
            ),
            "df_missing": df_missing,
            "html_missing": df_missing.to_html(
                index=False, classes=["table-wrapper2"]
            ),
            "df_unseen_cats": df_unseen_cats,
            "html_unseen_cats": df_unseen_cats.to_html(
                index=False, classes=["table-wrapper2"]
            ),
            "df_num_stats": df_num_stats,
            "html_num_stats": table_html,
            "df_type_issues": df_type_issues,
            "html_type_issues": df_type_issues.to_html(
                index=False, classes=["table-wrapper2"]
            ),
            "suggestions": suggestions,
            "samples_head": samples_head,
            "samples_middle": samples_middle,
            "samples_tail": samples_tail,
        }
    )
