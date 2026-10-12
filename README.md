# FLODIS — Multimodal Earth Observation for Socioeconomic Flood-Impact Prediction

Predicting the socioeconomic impacts of flood disasters (displacement, mortality, affected population, economic damage) by combining **Sentinel-1 radar**, **Sentinel-2 optical** imagery and **contextual data** (population, rainfall, terrain), using **pretrained deep-learning models** as feature extractors.

Internship project at **LIMOS — Université Clermont Auvergne**, within the **DLISCES** project (LIMOS × CERDI collaboration).
Supervisor: Julien Ah-Pine · Study period: flood events 2015–2018

---

## Overview

Satellite imagery shows *where* a flood happened, but not *what it cost* the people affected. This project builds a multimodal flood-event dataset linking reported socioeconomic impacts to satellite observations, then evaluates whether pretrained Earth Observation models can extract representations useful for event-level impact prediction.

### Pipeline

```
FLODIS socioeconomic records   (event IDs, dates, impact labels)
        │
        ▼
GFD flood footprints           (observed flooded areas, linked via GFD_ID)
        │
        ▼
Satellite observations         (Sentinel-1 & Sentinel-2, before/after the event)
  + Contextual data            (WorldPop, GPM IMERG, SRTM)
        │
        ▼
Pretrained feature extraction  (patch-level → aggregated per event)
        │
        ▼
Impact prediction              (Ridge, Random Forest, Extra Trees)
```

## Dataset construction

- **Labels:** FLODIS records (impacts from IDMC and EM-DAT). Missing labels are kept as missing — each target is evaluated on its own available subset.
- **Events:** 163 FLODIS–GFD records → **104 unique flood events** (2015–2018).
- **Flood footprints:** MODIS-derived Global Flood Database rasters (~250 m), vectorised in Google Earth Engine.
- **Sentinel-1 (GRD, IW, VV/VH):** 30-day before/after median composites + delta bands + flood mask, 10 m.
- **Sentinel-2 (B2, B3, B4, B8, B11, B12):** 60-day before/after median composites, cloud/shadow masking with SCL, valid-data masks, 10 m.
- **Patches:** 5.12 km × 5.12 km (512 × 512 px) grid tiles covering each flood footprint.
- **Context:** WorldPop (population exposure), GPM IMERG (rainfall), SRTM (elevation/terrain), reduced to event-level features.

> Patches from the same flood share one label and are **not independent observations**. All evaluation is done at the event level, with no patches from the same event split across train and test.

## Models

| Family | Model | Data |
|---|---|---|
| Flood-specific | SNUNet | Sentinel-1 |
| Flood-specific | FloodViT | Sentinel-1 |
| Flood-specific | sm_unet_s2 | Sentinel-2 |
| EO foundation model | CROMA | Radar & optical |
| EO foundation model | Clay | Sentinel-1 & Sentinel-2 |
| EO foundation model | DOFA | Sentinel-2 |

Models are used **frozen** as feature extractors. Patch features are aggregated per event (`GFD_ID`) and passed to classical regressors, which keeps the number of learned parameters small relative to the limited labelled data.

## Results

Displacement prediction, repeated cross-validation on a common subset of **45 events**:

| Representation | Regressor | Mean log RMSE ↓ | Mean R² ↑ |
|---|---|---|---|
| Clay S1 + DOFA S2 | Ridge | **2.452** | **0.154** |
| Clay S1 + Clay S2 | Extra Trees | 2.453 | 0.154 |
| DOFA S2 | Ridge | 2.468 | 0.144 |
| Clay S2 | Random Forest | 2.477 | 0.133 |

- Combining radar and optical representations gave a modest improvement over single-sensor representations.
- In a separate 101-event experiment, adding compact Sentinel-1 descriptors to WorldPop + GPM features improved over the contextual reference in **28 of 30** runs.

Results come from different experimental cohorts and should only be compared within the same cohort. Full outputs are in [`results/`](results/).

## Repository structure

```
├── code/
│   ├── (preprocessing scripts)
│   ├── Clay/  CROMA/  DOFA/  KuroSiwo/  UDL4FL/
│   └── impact_prediction/        # regression & cross-validation experiments
├── gee_scripts/                  # Google Earth Engine (JavaScript)
│   ├── 01_gfd_flood_polygons.js
│   ├── 02_sentinel1_all_patches.js
│   ├── 03_sentinel2_all_patches.js
│   ├── 04_worldpop.js
│   ├── 05_gpm_rainfall.js
│   ├── 06_srtm_elevation.js
│   └── ASSETS.md
└── results/                      # evaluation summaries & CV predictions (CSV)
```

## Data and reproducibility

Satellite patches, model checkpoints and full model repositories are **not included** because of their size. The scripts in `code/` are selected scripts and some depend on the original model repositories (Clay, CROMA, DOFA, etc.).

To reproduce the work you need:
- Google Earth Engine access (custom assets are listed in `gee_scripts/ASSETS.md`)
- The pretrained model repositories and checkpoints
- Python dependencies (see `requirements.txt`)

## Limitations

- Small number of independent labelled flood events
- Incomplete and uncertain impact records across sources
- Sentinel-2 cloud contamination; temporal mismatch between sensors
- Study period limited to 2015–2018 (end of the GFD product)
- Limited socioeconomic vulnerability variables (poverty, housing, infrastructure)

## Future directions

Extending the dataset beyond 2018 (EM-DAT / IDMC + Copernicus EMS flood maps), comparing labels with DFO records, adding richer socioeconomic data, joint radar–optical representation learning, and learned event-level aggregation (attention pooling / multiple-instance learning).

## Author

**Melissa Fahed** — AI/ML · Earth Observation
