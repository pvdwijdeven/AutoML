import pandas as pd
import numpy as np
import re
from typing import Tuple
from library import check_classification


def analyze_test_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame, target
) -> Tuple[str, str]:
    html_sections = []

    # 1. Size comparison
    rows_train, cols_train = df_train.shape
    rows_test, cols_test = df_test.shape
    size_table = (
        pd.DataFrame(
            {
                "Dataset": ["Train", "Test"],
                "Rows": [rows_train, rows_test],
                "Columns": [cols_train, cols_test],
            }
        )
        .to_html(index=False, classes=["frequency-table"])
        .replace('border="1"', "")
    )
    html_sections.append(
        f'<span id="size-comparison"><br></span><h3>Dataset Size Comparison</h3>{size_table}'
    )
    if rows_test < 0.5 * rows_train:
        html_sections.append(
            "<p><b>Suggestion:</b> The test set is much smaller than the training set. Results might be unstable or under-represented.</p>"
        )
    elif rows_test > 2 * rows_train:
        html_sections.append(
            "<p><b>Suggestion:</b> The test set is much larger than the training set. Consider whether the training data is representative enough.</p>"
        )

    # 2. Column comparison
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    only_in_train = sorted(train_cols - test_cols)
    only_in_test = sorted(test_cols - train_cols)
    col_check = pd.DataFrame(
        {
            "Only in Train": pd.Series(only_in_train).replace(
                target, f"{target} (target)"
            ),
            "Only in Test": pd.Series(only_in_test) if only_in_test else "None",
        }
    )
    html_sections.append(
        f'<span id="column-differences"><br></span><h3>Column Differences</h3>{col_check.to_html(index=False, classes=["frequency-table"]).replace('border="1"', "")}'
    )
    if len(only_in_train) > 0 or only_in_test:
        html_sections.append(
            "<p><b>Suggestion:</b> Ensure column alignment. Consider dropping or filling extra columns before prediction.</p>"
        )
    else:
        html_sections.append(
            "<p><b>Suggestion:</b> Columns align correctly between train and test.</p>"
        )

    # 3. Missing values
    missing_train = df_train.isna().mean().round(3) * 100
    missing_test = df_test.isna().mean().round(3) * 100
    missing_df = pd.DataFrame(
        {
            "Missing % (Train)": missing_train,
            "Missing % (Test)": missing_test,
        }
    ).fillna("-")
    html_sections.append(
        f'<span id="missing-values"><br></span><h3>Missing Value Comparison</h3>{missing_df.to_html(classes=["frequency-table"]).replace('border="1"', "")}'
    )
    test_missing_cols = missing_test[missing_test > 0].sort_values(
        ascending=False
    )
    if not test_missing_cols.empty:
        html_sections.append(
            "<p><b>Suggestion:</b> Handle missing values in test set — consider using imputation (mean, mode, model-based) or using 'unknown' category for missing categorical data.</p>"
        )

    # 4. Categorical: unseen categories
    cat_cols = []
    for col in df_train.columns:
        if check_classification(df_train[col], True):
            cat_cols.append(col)
    cat_cols = sorted(cat_cols, key=str.lower)
    unseen_data = []
    for col in cat_cols:
        if col in df_test.columns:
            train_cats = set(df_train[col].dropna().astype(str).unique())
            test_cats = set(df_test[col].dropna().astype(str).unique())
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
    if unseen_data:
        unseen_df = pd.DataFrame(unseen_data)
        html_sections.append(
            f'<span id="unseen-categories"><br></span><h3>Unseen Categories in Test Set</h3>{unseen_df.to_html(index=False,classes=["frequency-table"]).replace('border="1"', "")}'
        )
        html_sections.append(
            "<p><b>Suggestion:</b> Your model might fail or misinterpret unseen categories. Use encoders that support unknowns (e.g., `handle_unknown='ignore'` in sklearn’s OneHotEncoder or fallback labels in target encoding).</p>"
        )
    else:
        html_sections.append(
            '<span id="unseen-categories"><br></span><h3>Unseen Categories in Test Set</h3><p>None detected.</p>'
        )
        html_sections.append(
            "<p><b>Suggestion:</b> No unseen categories found — safe for categorical handling.</p>"
        )

    # 5. Numeric distributions (with mean/std/min/max shift detection and inline coloring)
    num_cols = sorted(
        df_train.select_dtypes(include=np.number).columns, key=str.lower
    )
    stats = []
    for col in num_cols:
        if col in df_test.columns:
            train_mean = df_train[col].mean()
            test_mean = df_test[col].mean()
            train_std = df_train[col].std()
            test_std = df_test[col].std()
            train_min = df_train[col].min()
            test_min = df_test[col].min()
            train_max = df_train[col].max()
            test_max = df_test[col].max()

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

    stats_df = pd.DataFrame(stats)

    def highlight_shifts(row):
        style = [""] * len(row)
        col_idx = list(row.index)
        if row["Mean Shift"]:
            style[col_idx.index("Test Mean")] = (
                "background-color: var(--bs-danger);color: var(--bs-light);"
            )
        if row["Std Shift"]:
            style[col_idx.index("Test Std")] = (
                "background-color: var(--bs-danger);color: var(--bs-light);"
            )
        if row["Min Shift"]:
            style[col_idx.index("Test Min")] = (
                "background-color: var(--bs-danger);color: var(--bs-light);"
            )
        if row["Max Shift"]:
            style[col_idx.index("Test Max")] = (
                "background-color: var(--bs-danger);color: var(--bs-light);"
            )
        return style

    styled = stats_df.style.apply(highlight_shifts, axis=1)
    styled = styled.hide(
        axis=1, subset=["Mean Shift", "Std Shift", "Min Shift", "Max Shift"]
    )

    # Generate raw HTML
    styled_html = (
        styled.hide(axis="index")
        .set_table_attributes('class="frequency-table"')
        .to_html()
        .replace('border="1"', "")
    )

    # Extract <style> block
    style_match = re.search(r"(<style.*?>.*?</style>)", styled_html, re.DOTALL)
    style_block = style_match.group(1) if style_match else ""

    # Remove the style from table HTML
    table_only = re.sub(
        r"<style.*?>.*?</style>", "", styled_html, flags=re.DOTALL
    )

    # Append HTML content (Jinja handles placement)
    html_sections.append(
        '<span id="numeric-comparison"><br></span><h3>Numeric Feature Distribution Comparison</h3>'
    )
    html_sections.append(table_only)
    html_sections.append(
        """
        <p><b>Suggestion:</b> Cells highlighted in light red indicate significant distribution shifts between training and test data.</p>
        <ul>
            <li><b>Mean/Std:</b> Could indicate changes in data patterns. Consider normalization or retraining.</li>
            <li><b>Min/Max:</b> Values fall outside expected range. May cause extrapolation errors for some models (e.g., tree splits, linear assumptions).</li>
        </ul>
    """
    )

    # 6. Data type mismatches
    dtype_issues = []
    for col in df_train.columns.intersection(df_test.columns):
        train_type = str(df_train[col].dtype)
        test_type = str(df_test[col].dtype)
        if train_type != test_type:
            dtype_issues.append(
                {
                    "Column": col,
                    "Train Dtype": train_type,
                    "Test Dtype": test_type,
                }
            )
    if dtype_issues:
        dtype_df = pd.DataFrame(dtype_issues)
        html_sections.append(
            f'<span id="dtype-mismatches"><br></span><h3>Data Type Mismatches</h3>{dtype_df.to_html(index=False,classes=["frequency-table"]).replace('border="1"', "")}'
        )
        html_sections.append(
            "<p><b>Suggestion:</b> Apply consistent dtype conversions. Data type mismatches may break transformation or prediction pipelines.</p>"
        )
    else:
        html_sections.append(
            '<span id="dtype-mismatches"><br></span><h3>Data Type Mismatches</h3><p>None found.</p>'
        )
        html_sections.append(
            "<p><b>Suggestion:</b> All columns have consistent data types.</p>"
        )

    return style_block, "\n".join(html_sections)
