// =====================================================
// PHASE 1 — WorldPop Population Exposure Features
// ONE FULL YEAR EXPORT
// Optimized: SCALE 2000 + simplify 5000
// Clean CSV, no geometry
// =====================================================


// ===============================
// 1. SETTINGS
// ===============================

var EVENTS_ASSET = "projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018";

var YEAR = 2016;

// Use full year first
var BATCH_START = 0;
var BATCH_END   = 999;

// Coarser scale for speed
var SCALE = 2000;

// Simplify large flood polygons
var SIMPLIFY_METERS = 5000;


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
// 3. LOAD WORLDPOP
// ===============================

var worldpop = ee.ImageCollection("WorldPop/GP/100m/pop");


// ===============================
// 4. GET WORLDPOP IMAGE FOR YEAR
// ===============================

function getWorldPopImage(year) {

  year = ee.Number(year);

  return worldpop
    .filter(ee.Filter.eq("year", year))
    .mosaic()
    .rename("population");
}


// ===============================
// 5. PROCESS EACH EVENT
// ===============================

function processEvent(event) {

  var geom = event.geometry();
  var simpleGeom = geom.simplify(SIMPLIFY_METERS);

  var year = ee.Number(event.get("year"));

  var popImg = getWorldPopImage(year);

  var reducer = ee.Reducer.sum()
    .combine(ee.Reducer.mean(), "", true)
    .combine(ee.Reducer.max(), "", true);

  var stats = popImg.reduceRegion({
    reducer: reducer,
    geometry: simpleGeom,
    scale: SCALE,
    maxPixels: 1e13,
    bestEffort: true,
    tileScale: 4
  });

  var popTotal = ee.Algorithms.If(
    ee.Algorithms.IsEqual(stats.get("population_sum"), null),
    -9999,
    stats.get("population_sum")
  );

  var popMean = ee.Algorithms.If(
    ee.Algorithms.IsEqual(stats.get("population_mean"), null),
    -9999,
    stats.get("population_mean")
  );

  var popMax = ee.Algorithms.If(
    ee.Algorithms.IsEqual(stats.get("population_max"), null),
    -9999,
    stats.get("population_max")
  );

  var gfdId = ee.Number(event.get("GFD_ID"));

  var row_id = ee.String(gfdId.toInt())
    .cat("_")
    .cat(ee.String(event.get("ISO3")));

  return ee.Feature(null, {
    row_id: row_id,
    GFD_ID: gfdId,
    ISO3: event.get("ISO3"),
    year: year,

    displacements: event.get("displacements"),
    flood_area_km2: event.get("flood_area_km2"),

    population_total: popTotal,
    population_mean: popMean,
    population_max: popMax
  });
}


// ===============================
// 6. RUN
// ===============================

var result = events.map(processEvent);

print("Result:", result);
print("First processed event:", result.first());


// ===============================
// 7. EXPORT
// ===============================

var fileName = ee.String("WorldPop_phase1_")
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
    "population_total",
    "population_mean",
    "population_max"
  ]
});