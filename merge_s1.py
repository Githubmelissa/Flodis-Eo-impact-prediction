import pandas as pd
import glob
import os

# =====================================================
# MERGE SENTINEL-1 PHASE 1 CSV FILES
# Keep ALL 163 rows — no duplicate removal
# =====================================================


# ===============================
# 1. FOLDER PATH
# ===============================

folder = r"/home/melissa/Downloads/Melissa_intern/Sentinel 1 features"


# ===============================
# 2. FIND CSV FILES
# ===============================

csv_files = glob.glob(
    os.path.join(folder, "S1_phase1*.csv")
)

# Exclude already-merged outputs if script is run again
csv_files = [
    f for f in csv_files
    if "S1_phase1_ALL" not in os.path.basename(f)
]

print("\nFound CSV files:")
for f in sorted(csv_files):
    print(os.path.basename(f))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {folder}")


# ===============================
# 3. READ CSV FILES
# ===============================

dfs = []

for file in sorted(csv_files):
    print(f"\nReading: {os.path.basename(file)}")
    df = pd.read_csv(file)
    dfs.append(df)


# ===============================
# 4. MERGE
# ===============================

merged = pd.concat(dfs, ignore_index=True)

print("\n===================================")
print("Merged shape:")
print(merged.shape)


# ===============================
# 5. FIX INVALID VALUES
# ===============================

# If Sentinel-1 count is 0, then SAR-derived flood fraction is invalid
merged.loc[
    merged["during_s1_count"] == 0,
    "flood_fraction_s1_proxy"
] = pd.NA


# ===============================
# 6. SAVE RAW MASTER DATASET
# ===============================

output_path = os.path.join(
    folder,
    "S1_phase1_ALL_RAW.csv"
)

merged.to_csv(output_path, index=False)

print("\n===================================")
print("FINAL RAW DATASET SAVED:")
print(output_path)
print("Rows:", len(merged))


# ===============================
# 7. VALIDATION
# ===============================

print("\n===================================")
print("Rows per year:")
print(merged.groupby("year").size())

print("\n===================================")
print("Missing values:")
print(merged.isna().sum())

print("\n===================================")
print("First rows:")
print(merged.head())

print("\n===================================")
print("Columns:")
print(list(merged.columns))