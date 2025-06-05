## Exercise 7 (10 minutes): Anomaly Detection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 1. Generate synthetic "normal" data
#    E.g., two features representing normal operating ranges (e.g., purchase amounts, usage rates, etc.)
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=(200, 2))  # 200 points around mean=50

# 2. Generate synthetic "anomalous" data
#    Points that deviate significantly from the normal distribution
outliers = np.array([[100, 100], [10, 90], [90, 10], [120, 40], [40, 120]])

# 3. Combine the datasets
X = np.vstack((normal_data, outliers))

# 4. Apply DBSCAN
#    eps controls the neighborhood radius; min_samples is how many samples must be within eps to form a cluster
db = DBSCAN(eps=10, min_samples=5)  # eps și min_samples sunt hiperparametri importanți
db.fit(X)

# 5. Identify outliers (DBSCAN labels them as -1)
labels = db.labels_  # Etichete de cluster
X_df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
X_df["cluster"] = labels

# 6. Visualization
plt.figure(figsize=(8, 6))
colors = np.array(['blue', 'green', 'red', 'purple', 'orange', 'gray'])

# Outlieri cu -1
plt.scatter(X_df["Feature1"], X_df["Feature2"], c=X_df["cluster"], cmap='tab10', s=50)
plt.title("DBSCAN Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# 7. Reporting
num_outliers = sum(labels == -1)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f"\nNumber of clusters found: {num_clusters}")
print(f"Number of anomalies detected: {num_outliers}")