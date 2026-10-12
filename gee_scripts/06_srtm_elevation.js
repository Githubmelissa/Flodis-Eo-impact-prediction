// =====================================================
// PHASE 1 — SRTM Terrain Features
// Elevation + Slope
// ONE FULL YEAR EXPORT
// Static context features, no before/after
// Clean CSV, no geometry
// =====================================================


// ===============================
// 1. SETTINGS
// ===============================

var EVENTS_ASSET = "projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018";

var YEAR = 2016;

// Use full year
var BATCH_START = 0;
var BATCH_END   = 999;

// SRTM native resolution is ~30m.
// For event-level features, 1000m is faster and acceptable.
var SCALE = 1000;

// Simplify large flood polygons
var SIMPLIFY_METERS = 2000;


// ===============================
// 2. LOAD EVENTS
// ===============================

var allEvents = ee.FeatureCollection(EVENTS_ASSET);

var allYearEvents = allEvents
  .filter(ee.Filter.eq("year", YEAR));

var events = ee.FeatureCollection(
  allYearEvents.toList(999).slice(BATCH_START, BATCH_END)
);

print("Events in year:", allYearEvents.size());
print("Selected events:", events.size());
print("First event:", events.first());


// ===============================
// 3. LOAD SRTM + SLOPE
// ===============================

var srtm = ee.Image("USGS/SRTMGL1_003")
  .select("elevation")
  .rename("elevation");

var slope = ee.Terrain.slope(srtm)
  .rename("slope");

var terrain = srtm.addBands(slope);


// ===============================
// 4. PROCESS EACH EVENT
// ===============================

function processEvent(event) {

  var geom = event.geometry();
  var simpleGeom = geom.simplify(SIMPLIFY_METERS);

  var gfdId = ee.Number(event.get("GFD_ID"));

  var row_id = ee.String(gfdId.toInt())
    .cat("_")
    .cat(ee.String(event.get("ISO3")));

  var reducer = ee.Reducer.mean()
    .combine(ee.Reducer.stdDev(), "", true)
    .combine(ee.Reducer.min(), "", true)
    .combine(ee.Reducer.max(), "", true);

  var stats = terrain.reduceRegion({
    reducer: reducer,
    geometry: simpleGeom,
    scale: SCALE,
    maxPixels: 1e13,
    bestEffort: true,
    tileScale: 4
  });

  function safeGet(key) {
    return ee.Algorithms.If(
      ee.Algorithms.IsEqual(stats.get(key), null),
      -9999,
      stats.get(key)
    );
  }

  return ee.Feature(null, {
    row_id: row_id,
    GFD_ID: gfdId,
    ISO3: event.get("ISO3"),
    year: event.get("year"),

    displacements: event.get("displacements"),
    flood_area_km2: event.get("flood_area_km2"),

    elevation_mean: safeGet("elevation_mean"),
    elevation_stdDev: safeGet("elevation_stdDev"),
    elevation_min: safeGet("elevation_min"),
    elevation_max: safeGet("elevation_max"),

    slope_mean: safeGet("slope_mean"),
    slope_stdDev: safeGet("slope_stdDev"),
    slope_min: safeGet("slope_min"),
    slope_max: safeGet("slope_max")
  });
}


// ===============================
// 5. RUN
// ===============================

var result = events.map(processEvent);

print("Result:", result);
print("First processed event:", result.first());


// ===============================
// 6. EXPORT
// ===============================

var fileName = ee.String("SRTM_phase1_")
  .cat(ee.Number(YEAR).format());

Export.table.toDrive({
  collection: result,
  description: fileName.getInfo(),
  fileNamePrefix: fileName.getInfo(),
  fileFormat: "CSV",
  selectors: [
    "row_id",
    "GFD_ID",
    "ISO3",
    "year",
    "displacements",
    "flood_area_km2",

    "elevation_mean",
    "elevation_stdDev",
    "elevation_min",
    "elevation_max",

    "slope_mean",
    "slope_stdDev",
    "slope_min",
    "slope_max"
  ]
});