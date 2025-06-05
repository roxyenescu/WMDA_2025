## Exercise 5 (10 minutes): Evaluating Clusters with Silhouette Scores
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load or assume you have a preprocessed dataset (df_scaled)
#    For demonstration, we'll again load & scale the Iris dataset
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit each clustering method
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Get the cluster labels from each method
labels_kmeans = kmeans.labels_
labels_dbscan = dbscan.labels_
labels_agg = agg.labels_

# 4. Compute silhouette scores (only if more than one cluster exists)
#    DBSCAN might produce a single cluster or no clusters if parameters are not well-tuned,
#    so we check to avoid an error in silhouette_score.
def safe_silhouette(X, labels, name):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        score = silhouette_score(X, labels)
        print(f"Silhouette score for {name}: {score:.3f}")
    else:
        print(f"Silhouette score for {name}: Not applicable (only 1 cluster detected)")

# 5. Print the scores
safe_silhouette(X_scaled, labels_kmeans, "KMeans")
safe_silhouette(X_scaled, labels_dbscan, "DBSCAN")
safe_silhouette(X_scaled, labels_agg, "Agglomerative")