#Develop a program to create histograms for all numerical features and analyze the distribution of each feature. 
#Generate box plots for all numerical features and identify any outliers. Use California Housing dataset.
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing

# -----------------------------
# Load California Housing Dataset
# -----------------------------
housing = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Set style
sns.set_style("whitegrid")

# -----------------------------
# HISTOGRAMS FOR ALL FEATURES
# -----------------------------
print("\nGenerating Histograms...\n")

df.hist(
    bins=30,
    figsize=(15, 10),
    edgecolor="black"
)

plt.suptitle(
    "Histograms of California Housing Numerical Features",
    fontsize=16
)

plt.tight_layout()
plt.show()

# -----------------------------
# BOX PLOTS FOR ALL FEATURES
# -----------------------------
print("\nGenerating Box Plots...\n")

plt.figure(figsize=(15, 10))

sns.boxplot(
    data=df
)

plt.title(
    "Box Plots of California Housing Numerical Features"
)

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -----------------------------
# OUTLIER DETECTION USING IQR
# -----------------------------
print("\nDetecting Outliers Using IQR Method...\n")

outlier_counts = {}

for col in df.columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    outlier_counts[col] = len(outliers)

# Print outlier counts
print("Outlier Counts Per Feature:\n")

for feature, count in outlier_counts.items():
    print(f"{feature}: {count}")

# -----------------------------
# INDIVIDUAL BOX PLOTS
# -----------------------------
print("\nGenerating Individual Box Plots...\n")

for col in df.columns:

    plt.figure(figsize=(6, 4))

    sns.boxplot(
        x=df[col]
    )

    plt.title(f"Boxplot of {col}")

    plt.tight_layout()

    plt.show()

print("\nAnalysis Completed Successfully.")