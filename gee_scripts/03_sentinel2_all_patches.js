// =====================================================
// FLODIS Event: Sentinel-2 ALL PATCHES EXPORT
// Grid-based version for ONE event
//
// This version:
// - does NOT randomly sample 50 patches
// - tiles the whole event polygon into 5120 m patches
// - exports all patches in batches
// - no visual layers
// - no explicit B11/B12 reproject
// - no valid-fraction reduceRegion in GEE
// - exports valid masks as bands for Python filtering later
//
// Exports 15 bands:
// 1  before_B2
// 2  before_B3
// 3  before_B4
// 4  before_B8
// 5  before_B11
// 6  before_B12
// 7  after_B2
// 8  after_B3
// 9  after_B4
// 10 after_B8
// 11 after_B11
// 12 after_B12
// 13 before_valid_mask
// 14 after_valid_mask
// 15 flood_mask
// =====================================================


// =====================================================
// 1. SETTINGS
// =====================================================

var EVENTS_ASSET = "projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018";

// Change this for each event
var GFD_ID = 4325;


// -----------------------------------------------------
// TEMPORAL WINDOW SETTINGS
// -----------------------------------------------------

var BEFORE_DAYS = 60;
var AFTER_DAYS = 60;

var S2_MODE = "SCL_fast_60_60_all_grid_patches";
var S2_FALLBACK_LEVEL = 1;


// -----------------------------------------------------
// PATCH SETTINGS
// -----------------------------------------------------

var PATCH_SIZE_M = 5120;
var SCALE = 10;

var DRIVE_FOLDER = "FLODIS_S2_ALL_PATCHES_ONE_EVENT";

// Batch export settings
// Example:
// first run:  BATCH_START = 0
// second run: BATCH_START = 100
// third run:  BATCH_START = 200
var BATCH_START = 1300;
var BATCH_SIZE = 50;

// Cloud filter
var MAX_CLOUDY_PIXEL_PERCENTAGE = 80;


// =====================================================
// 2. LOAD EVENT POLYGON
// =====================================================

var events = ee.FeatureCollection(EVENTS_ASSET);
var event = events.filter(ee.Filter.eq("GFD_ID", GFD_ID)).first();

print("Selected event:", event);

var roi = event.geometry();

print("Original ROI area km2:", roi.area(1).divide(1e6));

// Simplify large polygons to reduce computation
var roi_simple = ee.Algorithms.If(
  roi.area({maxError: 1}).gt(1e6),
  roi.simplify(2000),
  roi
);

roi_simple = ee.Geometry(roi_simple);

print("Simplified ROI area km2:", roi_simple.area(1).divide(1e6));


// =====================================================
// 3. GET EVENT DATE FROM GFD
// =====================================================

var gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1");
var gfd_event = ee.Image(gfd.filter(ee.Filter.eq("id", GFD_ID)).first());

var gfd_index = ee.String(gfd_event.get("system:index"));
var parts = gfd_index.split("_");

// Example:
// DFO_4580_From_20180305_to_20180321
var event_date = ee.Date.parse("YYYYMMdd", parts.get(3));

var before_start = event_date.advance(-BEFORE_DAYS, "day");
var before_end   = event_date.advance(-1, "day");

var after_start  = event_date;
var after_end    = event_date.advance(AFTER_DAYS, "day");

print("Event date:", event_date.format("YYYY-MM-dd"));
print("Before window:", before_start.format("YYYY-MM-dd"), "→", before_end.format("YYYY-MM-dd"));
print("After window:", after_start.format("YYYY-MM-dd"), "→", after_end.format("YYYY-MM-dd"));
print("S2 mode:", S2_MODE);


// =====================================================
// 4. LOAD SENTINEL-2 WITH FAST SCL MASK
// =====================================================

var s2sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(roi_simple)
  .filterDate(before_start, after_end)
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUDY_PIXEL_PERCENTAGE));

print("Raw S2 image count in before+after window:", s2sr.size());


// -----------------------------------------------------
// Fast SCL cloud mask
// -----------------------------------------------------
//
// Keep:
// 4 vegetation
// 5 bare soil
// 6 water
// 7 unclassified
//
// Remove:
// 0 no data
// 1 saturated/defective
// 2 dark pixels
// 3 cloud shadow
// 8 medium cloud
// 9 high cloud
// 10 cirrus
// 11 snow/ice
//
// B11/B12 are native 20 m.
// They are resampled implicitly at export because scale = 10.

function maskS2FastSCL(image) {
  var scl = image.select("SCL");

  var good = scl.eq(4)
    .or(scl.eq(5))
    .or(scl.eq(6))
    .or(scl.eq(7));

  var bands = image
    .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    .updateMask(good)
    .toInt16();

  return bands.copyProperties(image, image.propertyNames());
}

var s2 = s2sr.map(maskS2FastSCL);


// Split before/after
var s2_before = s2.filterDate(before_start, before_end);
var s2_after  = s2.filterDate(after_start, after_end);

print("S2 before image count:", s2_before.size());
print("S2 after image count:", s2_after.size());
print("If before or after image count is below 3, try 90/90 days.");


// =====================================================
// 5. CREATE BEFORE / AFTER COMPOSITES
// =====================================================

var before = s2_before.median()
  .rename([
    "before_B2",
    "before_B3",
    "before_B4",
    "before_B8",
    "before_B11",
    "before_B12"
  ])
  .toInt16();

var after = s2_after.median()
  .rename([
    "after_B2",
    "after_B3",
    "after_B4",
    "after_B8",
    "after_B11",
    "after_B12"
  ])
  .toInt16();


// =====================================================
// 6. VALID MASKS + FLOOD MASK
// =====================================================

var beforeValid = before.select("before_B4")
  .mask()
  .unmask(0)
  .rename("before_valid_mask")
  .toInt16();

var afterValid = after.select("after_B4")
  .mask()
  .unmask(0)
  .rename("after_valid_mask")
  .toInt16();

var floodMask = ee.Image.constant(1)
  .clip(roi_simple)
  .unmask(0)
  .rename("flood_mask")
  .toInt16();


// =====================================================
// 7. FINAL 15-BAND STACK
// =====================================================

var stack = before
  .addBands(after)
  .addBands(beforeValid)
  .addBands(afterValid)
  .addBands(floodMask)
  .toInt16();

print("Final S2 15-band stack:", stack.bandNames());


// =====================================================
// 8. CREATE ALL GRID PATCHES OVER THE EVENT
// =====================================================
//
// This replaces the old 50 random patch sampling.
// It creates a grid of 5120 m cells covering the flood polygon.
// Each grid cell becomes one patch.

var gridProjection = ee.Projection("EPSG:3857").atScale(PATCH_SIZE_M);

var grid = roi_simple.coveringGrid(gridProjection);

print("Total grid patches covering event:", grid.size());


// Add patch IDs and metadata
var totalPatchCount = grid.size();
var gridList = grid.toList(totalPatchCount);

var allPatches = ee.FeatureCollection(
  ee.List.sequence(0, totalPatchCount.subtract(1)).map(function(i) {
    i = ee.Number(i);

    var patch = ee.Feature(gridList.get(i));
    var patchGeom = patch.geometry();
    var patchArea = patchGeom.area(1);

    return patch.set({
      "patch_id": i,
      "GFD_ID": GFD_ID,
      "sensor": "S2",
      "patch_type": "event_grid_patch",

      // Tracking metadata
      "S2_mode": S2_MODE,
      "S2_fallback_level": S2_FALLBACK_LEVEL,
      "quality_strategy": "export_valid_masks_filter_later_in_python",

      // S2 image count metadata
      "raw_s2_count": s2sr.size(),
      "before_s2_count": s2_before.size(),
      "after_s2_count": s2_after.size(),

      // Method metadata
      "mask_method": "SCL_fast_keep_4_5_6_7",
      "resampling": "implicit_export_resampling_B11_B12_to_10m",
      "composite_method": "median",

      // Export metadata
      "bands_exported": "15 bands: 6 before, 6 after, before_valid_mask, after_valid_mask, flood_mask",
      "native_10m_bands": "B2,B3,B4,B8",
      "native_20m_bands_resampled_to_10m": "B11,B12",
      "values": "S2_SR_scaled_by_10000_Int16",

      // Patch metadata
      "patch_area_m2": patchArea,
      "patch_size_m": PATCH_SIZE_M,
      "flood_fraction": "compute_later_from_band_15",

      // Date metadata
      "event_date": event_date.format("YYYY-MM-dd"),
      "before_start": before_start.format("YYYY-MM-dd"),
      "before_end": before_end.format("YYYY-MM-dd"),
      "after_start": after_start.format("YYYY-MM-dd"),
      "after_end": after_end.format("YYYY-MM-dd"),
      "before_days": BEFORE_DAYS,
      "after_days": AFTER_DAYS,

      // Settings
      "max_cloudy_pixel_percentage": MAX_CLOUDY_PIXEL_PERCENTAGE,
      "export_scale_m": SCALE
    });
  })
);

print("All event grid patches:", allPatches);


// =====================================================
// 9. SELECT BATCH TO EXPORT
// =====================================================
//
// GEE cannot comfortably export hundreds of patches at once.
// Export in batches.
// Example:
// BATCH_START = 0,   BATCH_SIZE = 100
// BATCH_START = 100, BATCH_SIZE = 100
// BATCH_START = 200, BATCH_SIZE = 100

var batchList = allPatches.toList(BATCH_SIZE, BATCH_START);
var batchPatches = ee.FeatureCollection(batchList);

print("Exporting batch start:", BATCH_START);
print("Batch size requested:", BATCH_SIZE);
print("Batch patches:", batchPatches);


// =====================================================
// 10. EXPORT BATCH METADATA CSV
// =====================================================

Export.table.toDrive({
  collection: batchPatches,
  description: "S2_allpatches_metadata_event_" + GFD_ID + "_" + S2_MODE + "_batch_" + BATCH_START,
  folder: DRIVE_FOLDER,
  fileNamePrefix: "S2_allpatches_event_" + GFD_ID + "_" + S2_MODE + "_batch_" + BATCH_START + "_metadata",
  fileFormat: "CSV"
});


// =====================================================
// 11. EXPORT BATCH PATCH IMAGES
// =====================================================

batchPatches.evaluate(function(fc) {

  if (!fc || !fc.features) {
    print("ERROR: batchPatches did not evaluate. Try smaller BATCH_SIZE, for example 50.");
    return;
  }

  fc.features.forEach(function(f) {
    var patchGeom = ee.Geometry(f.geometry);
    var pid = f.properties.patch_id;

    var patchImage = stack.clip(patchGeom);

    Export.image.toDrive({
      image: patchImage,
      description: "S2_allpatches_event_" + GFD_ID + "_" + S2_MODE + "_patch_" + pid,
      folder: DRIVE_FOLDER,
      fileNamePrefix: "S2_allpatches_event_" + GFD_ID + "_" + S2_MODE + "_patch_" + pid,
      region: patchGeom,
      scale: SCALE,
      maxPixels: 1e9,
      fileFormat: "GeoTIFF"
    });
  });

  print("Submitted " + fc.features.length + " S2 all-patch export tasks for this batch. Go to Tasks tab → RUN.");
});