
# Google Earth Engine — Assets and Data Sources

**Project:** DLISCES — Flood Impact Prediction  
**Author:** Melissa Fahed  
**Supervisor:** Julien Ah-Pine  
**Laboratory:** LIMOS

## 1. Google Earth Engine Project

Project ID: `my-project1-453320`

Google Earth Engine Code Editor:
https://code.earthengine.google.com/

The JavaScript scripts used to construct the satellite and
contextual datasets are saved in the `gee_scripts/` folder.

## 2. Main FLODIS–GFD Asset

**Asset name:** FLODIS_ALL_GFD_polygons_2015_2018

**Asset ID:**
`projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018`

**Type:** FeatureCollection

**Study period:** 2015–2018

**Description:**

Flood polygons obtained from the Global Flood Database (GFD)
and linked to FLODIS socioeconomic impact records using GFD_ID.

This asset provides the flood geometries used to collect
Sentinel-1, Sentinel-2 and contextual data.

**Loading the asset in GEE:**

```javascript
var events = ee.FeatureCollection(
  'projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018'
);

print('Number of records:', events.size());

Map.centerObject(events);
Map.addLayer(events, {color: 'blue'}, 'Flood polygons');
```

## 3. Public Earth Engine Datasets

These datasets are accessed directly through GEE.
They do not need to be copied into the project assets.

| Dataset | GEE Collection ID |
|---|---|
| Global Flood Database | GLOBAL_FLOOD_DB/MODIS_EVENTS/V1 |
| Sentinel-1 | COPERNICUS/S1_GRD |
| Sentinel-2 | COPERNICUS/S2_SR_HARMONIZED |
| WorldPop | See WorldPop extraction script |
| GPM IMERG | See GPM extraction script |
| SRTM | See SRTM extraction script |

For WorldPop, GPM and SRTM, consult the saved JavaScript
files for the exact collections and products used.

## 4. Google Earth Engine Scripts

The `gee_scripts/` folder contains the code for:

- Extracting flood polygons from GFD
- Processing and exporting Sentinel-1 patches
- Processing and exporting Sentinel-2 patches
- Extracting WorldPop population information
- Extracting GPM precipitation information
- Extracting SRTM elevation and terrain information

The Sentinel-1 and Sentinel-2 scripts should use the
final full-coverage patch-extraction methodology.

## 5. Processed Data on the Server

The main project directory is:

`~/flood_project/`

**Sentinel-1 processed patches:**

`~/flood_project/S1_PATCHES_CLEAN/`

**Sentinel-2 processed patches:**

`~/flood_project/S2_PATCHES_CLEAN/`

**Additional Sentinel-2 exports:**

`~/flood_project/S2_RAW_EXPORTS/`

**Pretrained-model feature outputs:**

`~/flood_project/models_benchmark/`

**Regression results and predictions:**

`~/flood_project/models_benchmark/impact_prediction/outputs/`

The large datasets remain in their original server
directories and are not duplicated in FINAL_HANDOVER.

## 6. Other Custom Assets

Any additional custom assets referenced in the GEE
scripts should be documented here.

For each asset, record:
- Asset name
- Full Asset ID
- Purpose
- Scripts that depend on it

## 7. Access and Permissions

The flood-polygon asset is stored in the Google Earth
Engine project `my-project1-453320`.

Before running the scripts from another account, verify
that the researcher has the required permissions to
access the custom assets and use the GEE project.

Public Earth Engine collections can be loaded directly,
subject to the usual Earth Engine access requirements.
