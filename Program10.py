# Develop a program to implement k-means clustering using
# the Wisconsin Breast Cancer data set and visualize the clustering result.

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load and Scale Data
data = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(data.data)

# 2. Implement K-Means (k=2 since breast cancer is Malignant/Benign)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_scaled)
labels = kmeans.labels_

# 3. PCA for 2D Visualization
pca_data = PCA(n_components=2).fit_transform(X_scaled)

# 4. Plot Results
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title("K-Means Clustering (K=2)"); plt.legend(); plt.show()