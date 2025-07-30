import pandas as pd
import numpy as np
import plotly.express as px
from jinja2 import Environment, FileSystemLoader

# Sample data
np.random.seed(0)
df = pd.DataFrame(
    {
        "height": np.random.normal(170, 10, 100),
        "weight": np.random.normal(65, 12, 100),
    }
)


def get_frequency_table(df, column_name) -> str:
    frequency_data = df[column_name].value_counts().to_dict()
    if column_name == "Transported":
        print(set(frequency_data.keys()))
    if set(frequency_data.keys()) != {np.False_, np.True_}:
        frequency_data = {
            f'"{str(key)}"': value for key, value in frequency_data.items()
        }
    frequency_data["Missing values"] = df[column_name].isna().sum()
    total = len(df)

    frequency_table = """
    <table class="frequency-table" border="1">
    <thead>
        <tr>
        <th>Value</th>
        <th>Count</th>
        <th>Percentage</th>
        </tr>
    </thead>
    <tbody>
    """
    for key, value in frequency_data.items():
        percentage = value / total * 100
        frequency_table += f"""
        <tr>
        <td>{key}</td>
        <td>{value}</td>
        <td>{percentage:.1f}%</td>
        </tr>
        """

    frequency_table += """
    </tbody>
    </table>
    """

    return frequency_table


def get_html_from_template(template_file, context) -> str:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template(template_file)
    rendered_html = template.render(context=context)
    return rendered_html


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
