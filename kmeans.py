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
zones = zones.to_crs(epsg=4326)
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

    def calculate_distance(self, row):
        distances = []
        longitude = row["longitude"]
        latitude = row["latitude"]
        for i in range(len(self.centroids)):
            distances.append(np.sqrt((longitude-self.centroids[i][0])**2 + (latitude-self.centroids[i][1])**2))

        return distances

    def assign_cluster(self, row):
        return np.argmin(self.calculate_distance(row))

    def assign_all(self, data):
        self.labels = []
        for i in range(len(data)):
            row = data.iloc[i]
            self.labels.append(self.assign_cluster(row))

    def update_centroids(self, data):
        new_centroids = []

        for i in range(self.k):
            tot_lon = 0
            tot_lat = 0
            count = 0

            for j in range(len(data)):
                if self.labels[j] == i:
                    tot_lon += data.iloc[j]["longitude"]
                    tot_lat += data.iloc[j]["latitude"]
                    count += 1

            if count > 0:
                new_centroids.append([tot_lon / count, tot_lat / count])
            else:
                row = data.sample(1).iloc[0]
                new_centroids.append([row["longitude"], row["latitude"]])

        self.centroids = new_centroids

    def fit_model(self, data):
        self.initialize(data)

        for i in range(self.iterations):
            self.assign_all(data)
            self.update_centroids(data)

        return self.centroids, self.labels


kmeans = Kmeans(6, 10, 0 , 42)
centroids, labels = kmeans.fit_model(data)

data["cluster"] = labels

print("Cluster centers:")
for i, c in enumerate(centroids):
    print(f"Cluster {i}: Longitude={c[0]:.4f}, Latitude={c[1]:.4f}")

import matplotlib.pyplot as plt
zones = gpd.read_file("taxi_zones/taxi_zones.shp").to_crs(epsg=4326)


fig, ax = plt.subplots(figsize=(10, 10))
zones.boundary.plot(ax=ax, color="gray", linewidth=0.5)


plt.scatter(data["longitude"], data["latitude"], c=data["cluster"], cmap="tab10", s=5, alpha=0.6)


plt.scatter(
    [c[0] for c in centroids],
    [c[1] for c in centroids],
    c='black',
    marker='x',
    s=100,
    label='Centroids'
)

plt.title("NYC Taxi Pickup Clusters on Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.savefig("taxi_clusters_map.png", dpi=300)

print("Saved as taxi_clusters_map.png")
