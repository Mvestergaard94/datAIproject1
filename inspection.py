import pandas as pd

# Load your results
df = pd.read_csv("results/metrics.csv")

# 1) See the first few rows:
print(df.head())

# 2) Check for any missing values:
print("\nMissing values per column:\n", df.isnull().sum())

# 3) Get basic statistics on times and penalties:
print("\nSummary stats:\n", df.describe())
