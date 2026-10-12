// =====================================================
// PHASE 1 — GPM Rainfall Features
// ONE FULL YEAR EXPORT
// Clean CSV, no geometry
// =====================================================


// ===============================
// 1. SETTINGS
// ===============================

var EVENTS_ASSET = "projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018";

var GFD_ID = "GFD_ID";

var YEAR = 2017;

var DURING_DAYS = 15;

var RAIN_DAY_THRESHOLD_MM = 10;

var SCALE = 10000;


// ===============================
// 2. LOAD DATA
// ===============================

var gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1");

var gpm = ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
  .select("precipitationCal");

var allEvents = ee.FeatureCollection(EVENTS_ASSET);

var events = allEvents.filter(ee.Filter.eq("year", YEAR));

print("Events in year:", events.size());
print("First event:", events.first());


// ===============================
// 3. CONVERT GPM RATE TO MM
// ===============================

function toHalfHourlyMM(img) {
  return img
    .multiply(0.5)
    .rename("rain_mm")
    .copyProperties(img, ["system:time_start"]);
}


// ===============================
// 4. PROCESS EACH EVENT
// ===============================

function processEvent(event) {

  var geom = event.geometry();

  var simpleGeom = geom.simplify(5000);

  var gfdId = ee.Number(event.get(GFD_ID));

  var gfdEvent = ee.Image(
    gfd.filter(ee.Filter.eq("id", gfdId)).first()
  );

  var eventDate = ee.Date(
    gfdEvent.get("system:time_start")
  );

  var duringStart = eventDate;
  var duringEnd = eventDate.advance(DURING_DAYS, "day");

  var gpmCol = gpm
    .filterBounds(simpleGeom)
    .filterDate(duringStart, duringEnd)
    .map(toHalfHourlyMM);

  var gpmCount = gpmCol.size();

  var totalRainImg = ee.Image(
    ee.Algorithms.If(
      gpmCount.gt(0),
      gpmCol.sum().rename("rainfall_total_mm"),
      ee.Image.constant(-9999).rename("rainfall_total_mm")
    )
  );

  var maxRainImg = ee.Image(
    ee.Algorithms.If(
      gpmCount.gt(0),
      gpmCol.max().rename("rainfall_max_mm"),
      ee.Image.constant(-9999).rename("rainfall_max_mm")
    )
  );

  var meanRainImg = ee.Image(
    ee.Algorithms.If(
      gpmCount.gt(0),
      gpmCol.mean().rename("rainfall_mean_mm"),
      ee.Image.constant(-9999).rename("rainfall_mean_mm")
    )
  );

  var dayList = ee.List.sequence(0, DURING_DAYS - 1);

  var dailyRainCol = ee.ImageCollection(
    dayList.map(function(d) {

      d = ee.Number(d);

      var dayStart = duringStart.advance(d, "day");
      var dayEnd = dayStart.advance(1, "day");

      var dailyCol = gpm
        .filterBounds(simpleGeom)
        .filterDate(dayStart, dayEnd)
        .map(toHalfHourlyMM);

      var dailyImg = ee.Image(
        ee.Algorithms.If(
          dailyCol.size().gt(0),
          dailyCol.sum().rename("daily_rain_mm"),
          ee.Image.constant(0).rename("daily_rain_mm")
        )
      );

      return dailyImg.set("system:time_start", dayStart.millis());
    })
  );

  var rainyDaysImg = dailyRainCol
    .map(function(img) {
      return img.gt(RAIN_DAY_THRESHOLD_MM).rename("rainy_day");
    })
    .sum()
    .rename("rainfall_days");

  var rainfallFeaturesImg = totalRainImg
    .addBands(maxRainImg)
    .addBands(meanRainImg)
    .addBands(rainyDaysImg);

  var stats = rainfallFeaturesImg.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: simpleGeom,
    scale: SCALE,
    maxPixels: 1e9,
    bestEffort: true,
    tileScale: 4
  });

  var rainfallTotal = ee.Algorithms.If(
    ee.Algorithms.IsEqual(stats.get("rainfall_total_mm"), null),
    -9999,
    stats.get("rainfall_total_mm")
  );

  var rainfallMax = ee.Algorithms.If(
    ee.Algorithms.IsEqual(stats.get("rainfall_max_mm"), null),
    -9999,
    stats.get("rainfall_max_mm")
  );

  var rainfallMean = ee.Algorithms.If(
    ee.Algorithms.IsEqual(stats.get("rainfall_mean_mm"), null),
    -9999,
    stats.get("rainfall_mean_mm")
  );

  var rainfallDays = ee.Algorithms.If(
    ee.Algorithms.IsEqual(stats.get("rainfall_days"), null),
    -9999,
    stats.get("rainfall_days")
  );

  var row_id = ee.String(gfdId.toInt())
    .cat("_")
    .cat(ee.String(event.get("ISO3")));

  return ee.Feature(null, {
    row_id: row_id,
    GFD_ID: gfdId,
    ISO3: event.get("ISO3"),
    year: event.get("year"),

    displacements: event.get("displacements"),
    flood_area_km2: event.get("flood_area_km2"),

    gfd_event_date: eventDate.format("YYYY-MM-dd"),
    during_start: duringStart.format("YYYY-MM-dd"),
    during_end: duringEnd.format("YYYY-MM-dd"),

    gpm_count: gpmCount,

    rainfall_total_mm: rainfallTotal,
    rainfall_max_mm: rainfallMax,
    rainfall_mean_mm: rainfallMean,
    rainfall_days: rainfallDays
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

var fileName = ee.String("GPM_phase1_")
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
    "gfd_event_date",
    "during_start",
    "during_end",
    "gpm_count",
    "rainfall_total_mm",
    "rainfall_max_mm",
    "rainfall_mean_mm",
    "rainfall_days"
  ]
});