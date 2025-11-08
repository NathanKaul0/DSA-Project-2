import numpy as np
import pandas as pd
import geopandas as gpd
import requests

class DBSCAN:
    def __init__(self, radius=0.01, min_pts=5):
        self.radius = radius    # radius threshold
        self.min_pts = min_pts  # points required to label a region as "dense"
        self.labels = None      # cluster labels for each point
        self.n_clusters = 0

    def fit(self, data):        # convert to an array of numPy
        if isinstance(data, pd.DataFrame):
            self.X = data[["longitude", "latitude", "hour"]].to_numpy(dtype=float)
        else:
            self.X = np.array(data, dtype=float)

        n_points = len(self.X)      # number of samples
        self.labels = np.zeros(n_points, dtype=int)     # initialize to 0 indicate unvisited, then make it -1 if noise
        cluster_id = 0      # counter for clusters

        for i in range(n_points):
            if self.labels[i] != 0:
                continue    # if not 0 is visited

            neighbors = self.region_query(i)   # find all points within the radius of point i
            if len(neighbors) < self.min_pts:
                self.labels[i] = -1     # if it has fewer neighbors than the threshold, mark as noise
            else:
                cluster_id += 1     # else, it is a relevant point, add 1 to the counter of clusters
                self.expand_cluster(i, neighbors, cluster_id)      # and add the cluster

        self.n_clusters = cluster_id
        return self

    def region_query(self, point_index):     # find all neighbors within radius distance
        distances = np.linalg.norm(self.X - self.X[point_index], axis=1)    # np.linalg.norm is the Euclidean distance
        return np.where(distances <= self.radius)[0]    # returns an array of indices where distance is < radius

    def expand_cluster(self, point_index, neighbors, cluster_id):    # expand cluster recursively
        self.labels[point_index] = cluster_id
        i = 0       # iterator for the list of neighbors
        while i < len(neighbors):
            neighbor_index = neighbors[i]       # for each neighbor
            if self.labels[neighbor_index] == -1:   # if it was marked as noise
                self.labels[neighbor_index] = cluster_id    # relabeled it as member of the cluster
            elif self.labels[neighbor_index] == 0:  # if unvisited
                self.labels[neighbor_index] = cluster_id    # assign it to the cluster
                new_neighbors = self.region_query(neighbor_index)   # get all its neighbors
                if len(new_neighbors) >= self.min_pts:      # if the visited point is relevant
                    neighbors = np.append(neighbors, new_neighbors)     # append the neighbors to visit them and expand
            i += 1

    def fit_predict(self, data):    # just return the labels array to check if it works
        self.fit(data)
        return self.labels