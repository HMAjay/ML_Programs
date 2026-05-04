#Develop a program to implement Principal Component Analysis (PCA) for reducing the dimensionality 
#of the Iris dataset from 4 features to 2.
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load Iris Dataset
iris = load_iris()

# Features (X) and Target (y)
X = iris.data
y = iris.target

# Convert to DataFrame (optional but useful)
df = pd.DataFrame(X, columns=iris.feature_names)

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA(n_components=2)   # Reduce 4 features → 2 components
X_pca = pca.fit_transform(X_scaled)

# Step 4: Create DataFrame for PCA output
pca_df = pd.DataFrame(
    data=X_pca,
    columns=["Principal Component 1", "Principal Component 2"]
)

pca_df["Target"] = y

# Step 5: Visualize PCA result
plt.figure()

colors = ["red", "green", "blue"]

for i in range(3):
    plt.scatter(
        pca_df.loc[pca_df["Target"] == i,
                   "Principal Component 1"],
        pca_df.loc[pca_df["Target"] == i,
                   "Principal Component 2"],
        label=iris.target_names[i]
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset (4D → 2D)")
plt.legend()

plt.show()

# Step 6: Print Explained Variance Ratio
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)