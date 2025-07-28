import pandas as pd
import numpy as np
import plotly.express as px

# Sample data
np.random.seed(0)
df = pd.DataFrame(
    {
        "height": np.random.normal(170, 10, 100),
        "weight": np.random.normal(65, 12, 100),
    }
)


def create_plots_html(df):
    column_blocks = []

    for col in df.columns:
        # Horizontal box plot
        fig_box = px.box(
            df,
            x=col,
            title=f"{col} - Box Plot",
            orientation="h",
            height=300,
            width=600,
        )
        box_html = fig_box.to_html(full_html=False, include_plotlyjs="cdn")

        # Horizontal strip plot
        fig_strip = px.strip(
            df,
            x=col,
            title=f"{col} - Strip Plot",
            orientation="h",
            height=300,
            width=600,
        )
        strip_html = fig_strip.to_html(full_html=False, include_plotlyjs=False)

        # Text + Layout
        column_html = f"""
        <div style="margin-bottom: 60px;">
            <div class="row-flex">
                <!-- Text block -->
                <div class="block text-block">
                    <h3>{col}</h3>
                    <p>
                        <strong>Mean:</strong> {df[col].mean():.2f}<br>
                        <strong>Std:</strong> {df[col].std():.2f}<br>
                        <strong>Min:</strong> {df[col].min():.2f}<br>
                        <strong>Max:</strong> {df[col].max():.2f}
                    </p>
                </div>

                <!-- Box plot block -->
                <div class="block plot-block">
                    {box_html}
                </div>

                <!-- Strip plot block -->
                <div class="block plot-block">
                    {strip_html}
                </div>
            </div>
        </div>
        """

        column_blocks.append(column_html)

    return "\n".join(column_blocks)


# Generate content
content_html = create_plots_html(df)

HTML_HEAD = """<head>
    <title>Plots per Column</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .row-flex {
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
            align-items: flex-start;
        }
        .block {
            flex: 1 1 300px;
            box-sizing: border-box;
        }
        .text-block {
            max-width: 300px;
        }
        @media (max-width: 900px) {
            .row-flex {
                flex-direction: column;
            }
        }
    </style>
</head>"""

# Wrap into a full HTML document
FULL_HTML = f"""
<html>
{HTML_HEAD}
<body style="font-family: Arial, sans-serif; padding: 20px;">
    <h1>Data Overview</h1>
    {content_html}
</body>
</html>
"""

# Save to file
with open("plots_report.html", "w") as f:
    f.write(FULL_HTML)
