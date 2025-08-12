import pandas as pd


def summarize_results(results_dict):
    # Flatten nested dictionary into a list of rows
    rows = []
    for fold, models in results_dict.items():
        for model, scores in models.items():
            row = {"fold": fold, "model": model}
            row.update(scores)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate mean and std per model and score
    summary = df.groupby("model").agg(
        {
            col: ["mean", "std"]
            for col in df.columns
            if col not in ["fold", "model"]
        }
    )

    # Optional: flatten multi-index columns
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary.reset_index()


def write_to_output(output_file, total_result) -> None:
    summary_df = summarize_results(total_result)
    html_table: str = summary_df.to_html(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    with open(output_file, "w") as f:
        f.write(html_table)
