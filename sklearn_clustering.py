from sklearn.cluster import KMeans, MiniBatchKMeans

model = KMeans(n_clusters=3)

model = MiniBatchKMeans(n_clusters=3)

# (if we don't know how many cluster we're looking for, we can do a grid search on n_clusters)

# show the centroids coordinates

model.cluster_centers_

# show the cluster assigned to each observation
# (use this as hue parameter in sns scatterplot for example)

model.labels_
