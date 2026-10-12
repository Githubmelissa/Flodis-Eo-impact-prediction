
// 1. SETTINGS
var EVENTS_ASSET = "projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018";
var GFD_ID = 4465;  // change for each event
var BEFORE_DAYS = 30;
var AFTER_DAYS = 30;
var PATCH_SIZE_M = 5120;
var SCALE = 10;
var N_FINAL_FLOOD_PATCHES = 25;
var N_FINAL_CONTEXT_PATCHES = 25;
var N_FLOOD_CANDIDATES = 50;
var N_CONTEXT_CANDIDATES = 50;
var CONTEXT_BUFFER_M = 10000;
var SEED = 42;
var DRIVE_FOLDER = "FLODIS_S1_SAFE_PATCHES_GEE";
var MASK_SCALE = 500;

// 2. LOAD EVENT POLYGON
var events = ee.FeatureCollection(EVENTS_ASSET);
var event = events.filter(ee.Filter.eq("GFD_ID", GFD_ID)).first();
var roi = event.geometry();

// 2a. Safe simplification for large polygons
var roi_simple = ee.Algorithms.If(
  roi.area({maxError: 1}).gt(1e6),
  roi.simplify(2000),
  roi
);
roi_simple = ee.Geometry(roi_simple);

// 3. GET EVENT DATES
var gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1");
var gfd_event = ee.Image(gfd.filter(ee.Filter.eq("id", GFD_ID)).first());
var gfd_index = ee.String(gfd_event.get("system:index"));
var parts = gfd_index.split("_");
var event_date = ee.Date.parse("YYYYMMdd", parts.get(3));

var before_start = event_date.advance(-BEFORE_DAYS, "day");
var before_end   = event_date.advance(-1, "day");
var after_start  = event_date;
var after_end    = event_date.advance(AFTER_DAYS, "day");

print("Event date:", event_date.format("YYYY-MM-dd"));
print("Before window:", before_start.format("YYYY-MM-dd"), "→", before_end.format("YYYY-MM-dd"));
print("After window:", after_start.format("YYYY-MM-dd"), "→", after_end.format("YYYY-MM-dd"));

// 4. LOAD SENTINEL-1 STACK — all bands cast to Float32
var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(roi)
  .filter(ee.Filter.eq("instrumentMode", "IW"))
  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
  .select(["VV", "VH"]);

var s1_before = s1.filterDate(before_start, before_end);
var s1_after  = s1.filterDate(after_start,  after_end);

print("S1 before image count:", s1_before.size());
print("S1 after image count:",  s1_after.size());

var before  = s1_before.median().rename(["before_VV", "before_VH"]).toFloat();
var after   = s1_after.median().rename(["after_VV",  "after_VH"]).toFloat();
var deltaVV = after.select("after_VV").subtract(before.select("before_VV")).rename("delta_VV").toFloat();
var deltaVH = after.select("after_VH").subtract(before.select("before_VH")).rename("delta_VH").toFloat();
var floodMask = ee.Image.constant(1).clip(roi_simple).unmask(0).rename("flood_mask").toFloat();

var stack = before.addBands(after).addBands(deltaVV).addBands(deltaVH).addBands(floodMask);
// Bands: before_VV, before_VH, after_VV, after_VH, delta_VV, delta_VH, flood_mask (all Float32)

// 5. PATCH FUNCTIONS
var flood_area_image = floodMask.multiply(ee.Image.pixelArea());

function addFloodFraction(feature, patch_type) {
  var patchGeom = feature.geometry();
  var patchArea = ee.Number(patchGeom.area(1));  // FIX: 1m error margin
  var floodArea = ee.Number(
    flood_area_image.reduceRegion({
      reducer:   ee.Reducer.sum(),
      geometry:  patchGeom,
      scale:     MASK_SCALE,
      maxPixels: 1e8,
      bestEffort: true
    }).get("flood_mask")
  );
  floodArea = ee.Number(ee.Algorithms.If(floodArea, floodArea, 0));
  var floodFraction = floodArea.divide(patchArea).min(1);
  return feature.set({
    "GFD_ID":        GFD_ID,
    "patch_type":    patch_type,
    "patch_area_m2": patchArea,
    "flood_area_m2": floodArea,
    "flood_fraction": floodFraction
  });
}

function pointToPatch(feature, patch_type) {
  var geom = feature.geometry().buffer(PATCH_SIZE_M / 2 + 1).bounds();
  return addFloodFraction(ee.Feature(geom), patch_type);
}

function pointsToPatches(points, patch_type) {
  return points.map(function(f) { return pointToPatch(f, patch_type); });
}

// 6. SAMPLE PATCHES
var floodCandidates = pointsToPatches(
  ee.FeatureCollection.randomPoints(roi_simple, N_FLOOD_CANDIDATES, SEED),
  "flood_candidate"
).sort("flood_fraction", false).limit(N_FINAL_FLOOD_PATCHES);

var contextCandidates = pointsToPatches(
  ee.FeatureCollection.randomPoints(
    roi_simple.buffer(CONTEXT_BUFFER_M).bounds(),
    N_CONTEXT_CANDIDATES,
    SEED + 1
  ),
  "context_candidate"
).sort("flood_fraction", true).limit(N_FINAL_CONTEXT_PATCHES);

var finalPatchesRaw = floodCandidates.merge(contextCandidates);

// Add patch_id
var patchCount = N_FINAL_FLOOD_PATCHES + N_FINAL_CONTEXT_PATCHES;
var raw_patch_list = finalPatchesRaw.toList(patchCount);
var finalPatches = ee.FeatureCollection(
  ee.List.sequence(0, patchCount - 1).map(function(i) {
    return ee.Feature(raw_patch_list.get(i)).set("patch_id", i);
  })
);

print("Total patches to export:", finalPatches.size());

// 7. EXPORT METADATA CSV
Export.table.toDrive({
  collection:     finalPatches,
  description:    "metadata_event_" + GFD_ID,
  folder:         DRIVE_FOLDER,
  fileNamePrefix: "event_" + GFD_ID + "_metadata",
  fileFormat:     "CSV"
});

// 8. EXPORT PATCH IMAGES — use .evaluate() to avoid blocking .getInfo() in loop
finalPatches.evaluate(function(fc) {
  fc.features.forEach(function(f) {
    var patchGeom  = ee.Geometry(f.geometry);
    var pid        = f.properties.patch_id;
    var patchImage = stack.clip(patchGeom);

    Export.image.toDrive({
      image:          patchImage,
      description:    "event_" + GFD_ID + "_patch_" + pid,
      folder:         DRIVE_FOLDER,
      fileNamePrefix: "event_" + GFD_ID + "_patch_" + pid,
      region:         patchGeom,
      scale:          SCALE,
      maxPixels:      1e9,
      fileFormat:     "GeoTIFF"
    });
  });
  print("Submitted " + fc.features.length + " export tasks. Go to Tasks tab → click RUN ALL.");
});