# Develop a program to demonstrate the working of the decision tree algorithm. 
# Use Breast Cancer Data set for building the decision tree and apply this knowledge to classify a new sample.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load and Split Data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 2. Train Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

# 3. Predict and Evaluate
preds = dt.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")

# 4. Classify a New Sample (using the first test row as an example)
new_sample = X_test[0:1]
result = dt.predict(new_sample)
print(f"New Sample Classification: {data.target_names[result][0]}")

# 5. Visualize
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()