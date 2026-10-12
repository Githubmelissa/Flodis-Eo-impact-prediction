import pandas as pd

# Load original CSV
df = pd.read_csv("data/FLODIS_displacement.csv")

# Remove empty / unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df = df.loc[:, df.columns.notna()]
df = df.loc[:, df.columns.str.strip() != ""]

# Clean column names
df.columns = df.columns.str.strip()

# Save clean CSV
df.to_csv("FLODIS_displacement_clean.csv", index=False)

print("Clean file saved as FLODIS_displacement_clean.csv")
print("Columns:")
print(df.columns.tolist())
print("Rows:", len(df))