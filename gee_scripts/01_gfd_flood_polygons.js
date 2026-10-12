// =====================================================
// FLODIS + GFD — Extract GFD flood polygons
// Study window: 2015–2018
// NEXT BATCH: events 030 → 035
// =====================================================

var flodis = ee.FeatureCollection(
  "projects/my-project1-453320/assets/FLODIS_mortality_damage_GFD_ready"
);

var gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1");

var flodisStudy = flodis
  .filter(ee.Filter.notNull(["GFD_ID"]))
  .filter(ee.Filter.gte("year", 2015))
  .filter(ee.Filter.lte("year", 2018))
  .sort("event_id");

print("2015–2018 rows with GFD_ID:", flodisStudy.size());

var batchSize = 5;
var batchStart = 45;

var batchEnd = batchStart + batchSize;

var batchList = flodisStudy.toList(batchSize, batchStart);
var batch = ee.FeatureCollection(batchList);

print("Current batch start:", batchStart);
print("Current batch end:", batchEnd);
print("Current batch size:", batch.size());

var computeFeaturesWithPolygon = function(feature) {

  var gfd_id = ee.Number(feature.get("GFD_ID"));

  var flood = ee.Image(
    gfd.filter(ee.Filter.eq("id", gfd_id)).first()
  );

  var flooded = flood.select("flooded").eq(1).selfMask();

  var areaDict = flooded
    .multiply(ee.Image.pixelArea())
    .divide(1e6)
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: flood.geometry(),
      scale: 250,
      maxPixels: 1e13,
      bestEffort: true,
      tileScale: 8
    });

  var flood_area_km2 = ee.Number(areaDict.get("flooded"));

  var floodPolygons = flooded.reduceToVectors({
    geometry: flood.geometry(),
    scale: 250,
    geometryType: "polygon",
    eightConnected: true,
    maxPixels: 1e13,
    bestEffort: true,
    tileScale: 8
  });

  var floodGeometry = floodPolygons.geometry();

  var geometryAreaKm2 = floodGeometry.area(1).divide(1e6);

  return ee.Feature(floodGeometry, {
    event_id: feature.get("event_id"),
    GFD_ID: gfd_id,
    ISO3: feature.get("ISO3"),
    year: feature.get("year"),

    flood_area_km2: flood_area_km2,
    geometry_area_km2: geometryAreaKm2,

    total_deaths: feature.get("total_deaths"),
    no_affected_EMDAT: feature.get("no_affected_EMDAT"),
    total_affected_EMDAT: feature.get("total_affected_EMDAT"),
    total_damages_000_USD: feature.get("total_damages_(000_USD)")
  });
};

var resultBatch = batch.map(computeFeaturesWithPolygon);

print("Batch ready for export");

var firstPolygon = ee.Feature(resultBatch.first());

Map.centerObject(firstPolygon, 7);

Map.addLayer(
  firstPolygon,
  {color: "red"},
  "First polygon in current batch"
);

Export.table.toAsset({
  collection: resultBatch,
  description: "FLODIS_GFD_polygons_2015_2018_batch_045_050",
  assetId: "projects/my-project1-453320/assets/FLODIS_GFD_polygons_2015_2018_batch_045_050"
});