from jinja2 import Environment, FileSystemLoader
import plotly.express as px

# Create Plotly plots
df = px.data.iris()
fig1 = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig2 = px.histogram(df, x="petal_length", nbins=30)

plot_html_1 = fig1.to_html(full_html=False, include_plotlyjs=True)
plot_html_2 = fig2.to_html(full_html=False, include_plotlyjs=True)

# Prepare tab content
tabs = [
    {"title": "Overview", "content": plot_html_1},
    {"title": "Features", "content": plot_html_2},
    {"title": "Relations", "content": plot_html_2},
    {"title": "Missing values", "content": plot_html_2},
]

# Load and render the template
env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("report.html")
output_html = template.render(tabs=tabs)

# Save to file
with open("output.html", "w", encoding="utf-8") as f:
    f.write(output_html)

print("✅ HTML report generated: output.html")
