import pandas as pd
import geopandas as gpd
import requests


trips = pd.read_parquet(
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-07.parquet",
    columns=["tpep_pickup_datetime", "PULocationID"]
)


zones = gpd.read_file("taxi_zones/taxi_zones.shp")
print(zones.head())
zones["centroid"] = zones.geometry.centroid
zones["longitude"] = zones.centroid.x
zones["latitude"] = zones.centroid.y

zones = zones[["LocationID", "borough", "zone", "longitude", "latitude"]]
print(zones.head())


merged = trips.merge(zones, left_on="PULocationID", right_on="LocationID", how="left")
merged["hour"] = pd.to_datetime(merged["tpep_pickup_datetime"]).dt.hour


merged = merged.dropna(subset=["latitude", "longitude"])

sample_size = 100_000
data = merged.sample(sample_size, random_state=42)

print("Prepared data shape:", data.shape)
