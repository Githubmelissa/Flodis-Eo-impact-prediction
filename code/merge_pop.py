import pandas as pd
import glob
import os

# ==========================================
# Folder containing WorldPop CSVs
# ==========================================

folder = r"/home/melissa/Downloads/Melissa_intern/WorldPop_features"

# ==========================================
# Find CSV files
# ==========================================

csv_files = glob.glob(
    os.path.join(folder, "WorldPop_phase1_*.csv")
)

print("\nFound files:\n")

for f in sorted(csv_files):
    print(os.path.basename(f))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in:\n{folder}"
    )

# ==========================================
# Read all files
# ==========================================

dfs = []

for file in sorted(csv_files):

    print(f"\nReading: {os.path.basename(file)}")

    df = pd.read_csv(file)

    dfs.append(df)

# ==========================================
# Merge
# ==========================================

merged = pd.concat(dfs, ignore_index=True)

print("\n===================================")
print("Merged shape:")
print(merged.shape)

# ==========================================
# Save
# ==========================================

output_file = os.path.join(
    folder,
    "WorldPop_phase1_ALL.csv"
)

merged.to_csv(output_file, index=False)

print("\n===================================")
print("Saved:")
print(output_file)

print("\n===================================")
print("Rows per year:")
print(merged["year"].value_counts().sort_index())

print("\n===================================")
print("Missing values:")
print(merged.isna().sum())