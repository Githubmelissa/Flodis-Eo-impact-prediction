import ee
import geemap
import rasterio
import matplotlib.pyplot as plt

print("Packages imported successfully")

ee.Authenticate()
ee.Initialize(project="my-project1-453320")

EVENTS_ASSET = "projects/my-project1-453320/assets/FLODIS_ALL_GFD_polygons_2015_2018"

events = ee.FeatureCollection(EVENTS_ASSET)

first = ee.Feature(events.first())

print("Number of events:")
print(events.size().getInfo())

print("First event properties only:")
print(first.toDictionary().getInfo())