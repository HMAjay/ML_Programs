#Develop a program to compute the correlation matrix to understand the relationships between pairs of features. 
#Visualize the correlation matrix using a heatmap to know which variables have strong positive/negative correlations. 
#Create a pair plot to visualize pairwise relationships between features. 

# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import dataset
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
california = fetch_california_housing()

# Convert dataset into DataFrame
df = pd.DataFrame(
    data=california.data,
    columns=california.feature_names
)

# Add target column (House Value)
df["MedHouseVal"] = california.target


# -------------------------------
# 1. Compute Correlation Matrix
# -------------------------------

correlation_matrix = df.corr()

print("Correlation Matrix:\n")
print(correlation_matrix)


# -------------------------------
# 2. Visualize using Heatmap
# -------------------------------

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,        # Show correlation values
    cmap="coolwarm",   # Color theme
    fmt=".2f",         # Show values up to 2 decimal places
    linewidths=0.5     # Add grid lines
)

plt.title("Correlation Heatmap - California Housing Dataset")
plt.show()


# -------------------------------
# 3. Pair Plot Visualization
# -------------------------------

# Pair plot (use sample to reduce time)
sample_df = df.sample(n=500, random_state=42)

sns.pairplot(
    sample_df,
    diag_kind="hist"
)

plt.suptitle("Pair Plot of California Housing Features", y=1.02)

plt.show()