import ee
import pandas as pd

# -----------------------------
# 1. Initialize Earth Engine
# -----------------------------
try:
    ee.Initialize(project="my-project1-453320")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="my-project1-453320")


# -----------------------------
# 2. Load polygon asset
# -----------------------------
EVENTS_ASSET = "projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018"

events = ee.FeatureCollection(EVENTS_ASSET)

# -----------------------------
# 3. Get ALL GFD_IDs, including duplicates
# -----------------------------
all_ids = events.aggregate_array("GFD_ID").getInfo()

print("Total rows in asset:", len(all_ids))
print("All GFD_ID values, including duplicates:")
print(all_ids)

# -----------------------------
# 4. Save to CSV
# -----------------------------
df = pd.DataFrame({
    "row_index": range(len(all_ids)),
    "GFD_ID": all_ids
})

out_path = "all_gfd_ids_from_polygon_asset_with_duplicates.csv"
df.to_csv(out_path, index=False)

print("Saved:", out_path)

# -----------------------------
# 5. Optional: also show duplicate counts
# -----------------------------
counts = df["GFD_ID"].value_counts().reset_index()
counts.columns = ["GFD_ID", "count_in_asset"]

counts_path = "gfd_id_counts_in_polygon_asset.csv"
counts.to_csv(counts_path, index=False)

print("Saved:", counts_path)
print(counts)