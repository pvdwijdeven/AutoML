import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("data/train.csv")
profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile.to_file("data/report.html")
