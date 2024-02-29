import pandas as pd
data = pd.read_csv("data/census.csv")
df_clean = data.replace("?", None).dropna()
df_clean.to_csv("data/census_cleaned.csv", index=False)