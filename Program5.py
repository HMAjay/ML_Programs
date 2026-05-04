# Develop a program to implement k-Nearest Neighbour algorithm to classify the randomly generated 100 values of 
# x in the range of [0,1]. Perform the following based on dataset generated.
# 1. Label the first 50 points {x1,......,x50} as follows: if (xi ≤ 0.5), then xi ∊ Class1, else xi ∊ Class1
# 2. Classify the remaining points, x51,......,x100 using KNN. Perform this for k=1,2,3,4,5,20,30
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate 100 random values in [0,1]
np.random.seed(42)  # for reproducibility
x = np.random.rand(100)

# Step 2: Label first 50 points
labels = []

for i in range(50):
    if x[i] <= 0.5:
        labels.append("Class1")
    else:
        labels.append("Class2")

# Step 3: Prepare training data
X_train = x[:50].reshape(-1, 1)
y_train = np.array(labels)

# Step 4: Remaining points for testing
X_test = x[50:].reshape(-1, 1)

# Values of k
k_values = [1, 2, 3, 4, 5, 20, 30]

# Step 5: Apply KNN for different k values
for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    # Train model
    knn.fit(X_train, y_train)

    # Predict test points
    predictions = knn.predict(X_test)

    print("\n---------------------------")
    print(f"Results for k = {k}")
    print("---------------------------")

    for i in range(len(X_test)):
        print(
            f"x{50+i+1} = {X_test[i][0]:.3f} → {predictions[i]}"
        )