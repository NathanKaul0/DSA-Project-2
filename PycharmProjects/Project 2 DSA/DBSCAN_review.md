Run Down:

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together
points that are closely packed (dense regions) and marks points that lie alone in low-density areas as noise.

It requires two parameters:
- radius of the neighborhood around a point
- minimum number of points required to form a dense region

-> It starts declaring all the points as unvisited, so we can later either assign them a neighbor (Cluster ID) or declare them as noise
-> Then, we iterate over each point, and for every unvisited one, we will find all its neighbors
-> If the point meets the minimum number of neighbors requirement, we create a new cluster for it. Otherwise, it is marked as noise
-> If the cluster was created, we will then visit all the neighbor points of the cluster
-> This expansion will recursively visit all reachable points from the created cluster
-> The outer for loop will guarantee we visit each point in the data set, not only those within the range a newly created cluster


Time and Space Complexity:

Worst Case Time Complexity: Very dense data set, most points are relevant and few are noise
-> O(n^2) - for each point will process all the neighbors

Best Case Time Complexity: Very sparse data set, few clusters are created
-> O(n^2) - time complexity is still n^2 because the we iterate a subset of the n poins and calculate the distance to the other n points...
... however, most points will be marked as noise in the first pass and will be skipped in subsequent calls of region_query

Average Case Time Complexity: As a consequence, the average time complexity will be in the order of O(n^2)...
... more specifically, it will be O(n * m) where m is the average size of the neighborhoods


Space Complexity:

-> The space complexity is determined by the size n of the data set. The numPy array will have size n * 3 (longitude, latitude, hour)
-> For the code, there are at most n clusters (each point is relevant), so region_query will create at most n neighborhoods...
... which only adds n, making it O(n * 4), which translates into O(n) space complexity


Final Thoughts:

The Average Time Complexity of region_query could be improved using K-dimensional trees to make it O(n * log n)
However, using numPy already improves the space complexity, and its built-in properties are optimized for mathematical operations