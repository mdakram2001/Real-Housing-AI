import pandas as pd
from ydata_profiling import ProfileReport

# Load the cleaned dataset
df = pd.read_csv('my_gurgaon_properties_cleaned_v2.csv')

# Create the Pandas Profiling report
profile = ProfileReport(df, title="Gurgaon Properties EDA Report", explorative=True)

# Generate the report and save it as an HTML file
profile.to_file("gurgaon_properties_eda_report.html")