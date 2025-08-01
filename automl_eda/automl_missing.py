import pandas as pd


def missing_data_summary(df):
    # Count of missing values per column
    missing_count = df.isnull().sum()

    # Percentage of missing values per column
    missing_percentage = (missing_count / len(df)) * 100

    # Combine into a DataFrame and sort descending
    column_info = pd.DataFrame(
        {
            "Missing Count": missing_count.astype(str) + " of " + str(len(df)),
            "Missing Percentage": missing_percentage.map("{:.2f}%".format),
        }
    ).sort_values(by="Missing Count", ascending=False)

    # General missing data statistics
    total_missing = missing_count.sum()
    total_missing_text = f"{total_missing} of {len(df) * len(df.columns)} ({total_missing / (len(df) * len(df.columns)):.2%})"
    rows_with_missing = df.isnull().any(axis=1).sum()
    rows_with_missing_text = (
        f"{rows_with_missing} of {len(df)} ({rows_with_missing / len(df):.2%})"
    )
    columns_all_missing = (missing_count == len(df)).sum()
    columns_all_missing_text = f"{columns_all_missing} of {len(df.columns)} ({columns_all_missing / len(df.columns):.2%})"

    general_info = pd.DataFrame(
        {
            "Metric": [
                "Total Missing Values",
                "Rows with Missing Values",
                "Columns with 100% Missing",
            ],
            "Value": [
                total_missing_text,
                rows_with_missing_text,
                columns_all_missing_text,
            ],
        }
    )

    # Convert to HTML
    column_info_html = column_info.to_html(classes=["frequency-table"])
    general_info_html = general_info.to_html(header=False, index=False)

    return column_info_html, general_info_html
