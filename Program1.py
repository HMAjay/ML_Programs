#Develop a program to create histograms for all numerical features and analyze the distribution of each feature. 
#Generate box plots for all numerical features and identify any outliers. Use California Housing dataset.

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("housing.csv")

# Select only numerical columns
num_df = df.select_dtypes(include='number')

# Histograms
num_df.hist(bins=30)
plt.suptitle("Histograms")
plt.tight_layout()
plt.show()

# Box plots
num_df.plot(kind='box', subplots=True, layout=(3, 3), figsize=(12, 8))
plt.suptitle("Box Plots")
plt.tight_layout()
plt.show()

# Outlier detection (IQR)
print("Outliers count:")
Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).sum()