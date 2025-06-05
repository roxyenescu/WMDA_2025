## Exercise 3 (10 minutes): DBSCAN Clustering
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1. Assume df_scaled is the preprocessed DataFrame from Exercise 1
#    For demonstration, we'll again simulate df_scaled with the Iris dataset's features.
from sklearn.datasets import load_iris

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------

# 2. Instantiate DBSCAN with chosen parameters
#    eps defines the neighborhood radius, min_samples is the minimum number of points
#    for a region to be considered dense.
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 3. Fit the model to the data
dbscan.fit(df_scaled)

# 4. Extract cluster labels
cluster_labels = dbscan.labels_

# 5. Identify outliers (DBSCAN labels outliers as -1)
outliers = df_scaled[cluster_labels == -1]

# 6. (Optional) Add the labels to the DataFrame
df_scaled["cluster"] = cluster_labels

# 7. Print the cluster label counts
print("Distribuție clustere (incluzând -1 pentru outlieri):")
print(df_scaled["cluster"].value_counts())

# 8. Optional quick visualization (for 2D only)
#    Choose two features to plot, coloring by DBSCAN labels
plt.figure(figsize=(8, 6))
plt.scatter(
    df_scaled.iloc[:, 0],  # de exemplu: sepal length
    df_scaled.iloc[:, 1],  # de exemplu: sepal width
    c=df_scaled["cluster"],
    cmap="plasma",
    s=60,
    edgecolor='k'
)
plt.title("DBSCAN Clustering on Iris Dataset (2D View)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.grid(True)
plt.show()