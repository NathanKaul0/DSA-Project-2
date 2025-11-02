import numpy as np
import pandas as pd
import geopandas as gpd
import requests


print("Loading TLC Yellow Taxi Trip Data...")
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

sample_size = 50_000
data = merged.sample(sample_size, random_state=42)

print("Prepared data shape:", data.shape)


class Kmeans:
    def __init__(self, k, iterations, min, random_state = None):
        self.k = k
        self.iterations = iterations
        self.min = min
        self.random_state = random_state
        self.centroids = []
        self.labels = None

    def initialize(self, data):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        random_index = np.random.choice(len(data), size = self.k, replace=False)
        self.centroids = []
        for x in random_index:
            row = data.iloc[x]
            longitude = row["longitude"]
            latitude = row["latitude"]
            self.centroids.append([longitude, latitude])

        return self.centroids

    def calculate_distance(self, centroids):
        distances = []




