import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium


class Kmeans:
    def __init__(self, k, iterations, min, random_state=None):
        self.k = k
        self.iterations = iterations
        self.min = min
        self.random_state = random_state
        self.centroids = []
        self.labels = None

    def initialize(self, data):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        random_index = np.random.choice(len(data), size=self.k, replace=False)
        self.centroids = []
        for x in random_index:
            row = data.iloc[x]
            self.centroids.append([row["longitude"], row["latitude"]])
        return self.centroids

    def calculate_distance(self, row):
        distances = []
        for i in range(len(self.centroids)):
            distances.append(np.sqrt(
                (row["longitude"] - self.centroids[i][0]) ** 2 +
                (row["latitude"] - self.centroids[i][1]) ** 2
            ))
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
            tot_lon, tot_lat, count = 0, 0, 0
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
        for _ in range(self.iterations):
            self.assign_all(data)
            self.update_centroids(data)
        return self.centroids, self.labels



class DBSCAN:
    def __init__(self, radius=0.01, min_pts=5):
        self.radius = radius
        self.min_pts = min_pts
        self.labels = None
        self.n_clusters = 0

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.X = data[["longitude", "latitude", "hour"]].to_numpy(dtype=float)
        else:
            self.X = np.array(data, dtype=float)

        n_points = len(self.X)
        self.labels = np.zeros(n_points, dtype=int)
        cluster_id = 0

        for i in range(n_points):
            if self.labels[i] != 0:
                continue
            neighbors = self.region_query(i)
            if len(neighbors) < self.min_pts:
                self.labels[i] = -1
            else:
                cluster_id += 1
                self.expand_cluster(i, neighbors, cluster_id)

        self.n_clusters = cluster_id
        return self

    def region_query(self, point_index):
        distances = np.linalg.norm(self.X - self.X[point_index], axis=1)
        return np.where(distances <= self.radius)[0]

    def expand_cluster(self, point_index, neighbors, cluster_id):
        self.labels[point_index] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_index = neighbors[i]
            if self.labels[neighbor_index] == -1:
                self.labels[neighbor_index] = cluster_id
            elif self.labels[neighbor_index] == 0:
                self.labels[neighbor_index] = cluster_id
                new_neighbors = self.region_query(neighbor_index)
                if len(new_neighbors) >= self.min_pts:
                    neighbors = np.append(neighbors, new_neighbors)
            i += 1

    def fit_predict(self, data):
        self.fit(data)
        return self.labels



@st.cache_data(show_spinner=False)
def load_data():
    zones = gpd.read_file("taxi_zones/taxi_zones.shp").to_crs(epsg=4326)
    zones["centroid"] = zones.geometry.centroid
    zones["longitude"] = zones.centroid.x
    zones["latitude"] = zones.centroid.y

    trips = pd.read_parquet(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-07.parquet",
        columns=["tpep_pickup_datetime", "PULocationID"]
    )

    merged = trips.merge(
        zones[["LocationID", "longitude", "latitude"]],
        left_on="PULocationID", right_on="LocationID", how="left"
    )
    merged = merged.dropna(subset=["longitude", "latitude"])


    merged["hour"] = pd.to_datetime(merged["tpep_pickup_datetime"]).dt.hour
    return merged


data = load_data()
data_sample = data.sample(2000, random_state=42)
coords = data_sample[["longitude", "latitude", "hour"]].reset_index(drop=True)


kmeans_model = Kmeans(k=5, iterations=10, min=0, random_state=42)
kmeans_centroids, kmeans_labels = kmeans_model.fit_model(
    data_sample[["longitude", "latitude"]]
)
coords["kmeans_cluster"] = kmeans_labels


dbscan_model = DBSCAN(radius=0.01, min_pts=5)
dbscan_labels = dbscan_model.fit_predict(coords)
coords["dbscan_cluster"] = dbscan_labels


dbscan_centroids = []
for cluster_id in range(1, dbscan_model.n_clusters + 1):
    cluster_points = coords[coords["dbscan_cluster"] == cluster_id]
    if not cluster_points.empty:
        dbscan_centroids.append([
            cluster_points["longitude"].mean(),
            cluster_points["latitude"].mean()
        ])


st.title("🚕 NYC Taxi Pickup Clustering Comparison")
st.markdown("Comparing **K-Means** vs **DBSCAN** using custom implementations")

col1, col2 = st.columns(2)
center = [coords["latitude"].mean(), coords["longitude"].mean()]


with col1:
    st.subheader("K-Means")
    m1 = folium.Map(location=center, zoom_start=11, tiles="cartodb positron")

    for cluster_id, group in coords.groupby("kmeans_cluster"):
        color = f"#{np.random.randint(0, 0xFFFFFF):06x}"
        for _, row in group.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.6
            ).add_to(m1)

    for c in kmeans_centroids:
        folium.Marker(
            location=[c[1], c[0]],
            icon=folium.Icon(color="red", icon="star")
        ).add_to(m1)

    st_folium(m1, width=500, height=500)


with col2:
    st.subheader("DBSCAN")
    m2 = folium.Map(location=center, zoom_start=11, tiles="cartodb positron")

    for cluster_id, group in coords.groupby("dbscan_cluster"):
        color = f"#{np.random.randint(0, 0xFFFFFF):06x}"
        for _, row in group.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=2,
                color=color,
                fill=True,
                fill_opacity=0.6
            ).add_to(m2)

    for c in dbscan_centroids:
        folium.Marker(
            location=[c[1], c[0]],
            icon=folium.Icon(color="blue", icon="star")
        ).add_to(m2)

    st_folium(m2, width=500, height=500)
