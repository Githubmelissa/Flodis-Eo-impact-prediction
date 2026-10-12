import ee
import geemap
import os
import csv

# =====================================================
# SIMPLE PYTHON SCRIPT — Export 50 Sentinel-1 patches at 10 m
# No flood_fraction ranking
# Only patches that intersect the flood polygon
# =====================================================

ee.Initialize(project="my-project1-453320")

# =====================================================
# SETTINGS
# =====================================================
EVENTS_ASSET = "projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018"

GFD_ID_TO_TEST = 4584

PATCH_SIZE_M = 5120      # 5120 m ≈ 512 x 512 pixels at 10 m
EXPORT_SCALE = 10        # Sentinel-1 10 m export
MAX_PATCHES = 50         # try 50 patches

OUT_ROOT = "simple_50_s1_patch_exports_10m"


# =====================================================
# LOAD EVENT
# =====================================================
events = ee.FeatureCollection(EVENTS_ASSET)

event = ee.Feature(
    events
    .filter(ee.Filter.eq("GFD_ID", GFD_ID_TO_TEST))
    .first()
)

roi = event.geometry()
props = event.toDictionary().getInfo()
gfd_id = props["GFD_ID"]

print("\nSelected event:")
print(props)


# =====================================================
# GET REAL GFD EVENT DATE
# =====================================================
gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")

gfd_event_collection = gfd.filter(ee.Filter.eq("id", gfd_id))
gfd_count = gfd_event_collection.size().getInfo()

print("\nMatching GFD events found:", gfd_count)

if gfd_count == 0:
    raise ValueError(f"No GFD event found with id={gfd_id}")

gfd_event = ee.Image(gfd_event_collection.first())

flood_start = gfd_event.date()

before_start = flood_start.advance(-30, "day")
before_end = flood_start.advance(-1, "day")

after_start = flood_start
after_end = flood_start.advance(30, "day")

before_start_str = before_start.format("YYYY-MM-dd").getInfo()
before_end_str = before_end.format("YYYY-MM-dd").getInfo()
after_start_str = after_start.format("YYYY-MM-dd").getInfo()
after_end_str = after_end.format("YYYY-MM-dd").getInfo()
flood_start_str = flood_start.format("YYYY-MM-dd").getInfo()

print("\nGFD event date:", flood_start_str)
print("Before window:", before_start_str, "to", before_end_str)
print("After/during window:", after_start_str, "to", after_end_str)


# =====================================================
# CREATE SIMPLE PATCH GRID
# =====================================================
def create_simple_intersecting_patches(roi, patch_size_m, max_patches):
    """
    Create a grid over the polygon bounds.
    Keep only cells that intersect/touch the flood polygon.
    Do NOT calculate flood_fraction.
    This is much faster.
    """

    proj = ee.Projection("EPSG:3857").atScale(patch_size_m)

    grid = roi.bounds(maxError=10).coveringGrid(proj)

    patches = (
        grid
        .filterBounds(roi)
        .limit(max_patches)
    )

    return patches


print("\nCreating simple intersecting patches...")

patches = create_simple_intersecting_patches(
    roi=roi,
    patch_size_m=PATCH_SIZE_M,
    max_patches=MAX_PATCHES
)

num_patches = patches.size().getInfo()

print("\nNumber of selected patches:", num_patches)

if num_patches == 0:
    raise ValueError("No patches found.")

patch_list = patches.toList(num_patches)


# =====================================================
# SENTINEL-1 FUNCTION
# =====================================================
def get_s1_median(region, start_date, end_date):
    """
    Get Sentinel-1 VV median image for one patch.
    """

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select(["VV"])
    )

    count = s1.size().getInfo()

    if count == 0:
        return None, 0

    image = s1.median().clip(region)
    return image, count


# =====================================================
# EXPORT PATCHES
# =====================================================
event_dir = f"{OUT_ROOT}/event_{gfd_id}"
os.makedirs(event_dir, exist_ok=True)

metadata_path = f"{event_dir}/patch_metadata.csv"

with open(metadata_path, mode="w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "gfd_id",
        "patch_id",
        "before_start",
        "before_end",
        "after_start",
        "after_end",
        "s1_before_count",
        "s1_after_count",
        "exported"
    ])

    for i in range(num_patches):
        patch_feature = ee.Feature(patch_list.get(i))
        patch_geom = patch_feature.geometry()

        patch_id = f"patch_{i:03d}"
        patch_dir = f"{event_dir}/{patch_id}"
        os.makedirs(patch_dir, exist_ok=True)

        before_file = f"{patch_dir}/S1_before.tif"
        after_file = f"{patch_dir}/S1_after.tif"

        print("\n====================================")
        print(f"Processing {patch_id} / {num_patches}")
        print("====================================")

        # If you restart the script, skip already exported patches
        if os.path.exists(before_file) and os.path.exists(after_file):
            print("Already exported. Skipping.")

            writer.writerow([
                gfd_id,
                patch_id,
                before_start_str,
                before_end_str,
                after_start_str,
                after_end_str,
                "already_exists",
                "already_exists",
                True
            ])

            continue

        s1_before, before_count = get_s1_median(
            patch_geom,
            before_start_str,
            before_end_str
        )

        s1_after, after_count = get_s1_median(
            patch_geom,
            after_start_str,
            after_end_str
        )

        print("S1 before count:", before_count)
        print("S1 after count:", after_count)

        if s1_before is None or s1_after is None:
            print("Skipping patch because before or after Sentinel-1 image is missing.")

            writer.writerow([
                gfd_id,
                patch_id,
                before_start_str,
                before_end_str,
                after_start_str,
                after_end_str,
                before_count,
                after_count,
                False
            ])

            continue

        print("Exporting before image at 10 m...")
        geemap.ee_export_image(
            s1_before,
            filename=before_file,
            scale=EXPORT_SCALE,
            region=patch_geom,
            file_per_band=False
        )

        print("Exporting after image at 10 m...")
        geemap.ee_export_image(
            s1_after,
            filename=after_file,
            scale=EXPORT_SCALE,
            region=patch_geom,
            file_per_band=False
        )

        writer.writerow([
            gfd_id,
            patch_id,
            before_start_str,
            before_end_str,
            after_start_str,
            after_end_str,
            before_count,
            after_count,
            True
        ])

        print(f"Saved {patch_id}")


print("\nFinished.")
print("Output folder:", event_dir)
print("Metadata file:", metadata_path)