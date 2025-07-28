from jinja2 import Environment, FileSystemLoader
import plotly.express as px

# Create Plotly plots
df = px.data.iris()
fig1 = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig2 = px.histogram(df, x="petal_length", nbins=30)

plot_html_1 = fig1.to_html(full_html=False, include_plotlyjs=True)
plot_html_2 = fig2.to_html(full_html=False, include_plotlyjs=True)
col = "sepal_width"
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
            </div>
        </div>
        """


col = "petal_length"
column_html2 = f"""
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
            </div>
        </div>
        """
# Prepare tab content
tabs = [
    {"title": "Overview", "content": plot_html_1},
    {"title": "Features", "content": plot_html_2},
    {"title": "Relations", "content": column_html},
    {"title": "Missing values", "content": column_html2},
]

# Load and render the template
env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("EDA_report.html")
output_html = template.render(tabs=tabs, title="Iris Dataset EDA Report")

# Save to file
with open("output\\output.html", "w", encoding="utf-8") as f:
    f.write(output_html)

print("✅ HTML report generated: output.html")
