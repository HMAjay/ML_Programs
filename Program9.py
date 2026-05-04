# Develop a program to implement the Naive Bayesian classifier considering Olivetti Face Data set for training. 
# Compute the accuracy of the classifier, considering a few test data sets.

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and Split Data
faces = fetch_olivetti_faces()
xtr, xte, ytr, yte = train_test_split(faces.data, faces.target, test_size=0.2, random_state=42)

# 2. Train Gaussian Naive Bayes
model = GaussianNB().fit(xtr, ytr)

# 3. Predict and Evaluate
ypred = model.predict(xte)
print(f"Accuracy: {accuracy_score(yte, ypred):.2%}")
print("\nClassification Report:\n", classification_report(yte, ypred))