# Develop a program to demonstrate the working of Linear Regression and Polynomial Regression. 
# Use Boston Housing Dataset for Linear Regression and
# Auto MPG Dataset (for vehicle fuel efficiency prediction) for Polynomial Regression.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# --- 1. Linear Regression (Boston Housing) ---
url_b = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df_b = pd.read_csv(url_b)
X, y = df_b.drop('medv', axis=1), df_b['medv']

xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(xtr, ytr)
print(f"Linear Regression R2: {r2_score(yte, model.predict(xte)):.2f}")

# --- 2. Polynomial Regression (Auto MPG) ---
url_a = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
cols = ['mpg', 'cyl', 'disp', 'hp', 'wt', 'acc', 'yr', 'orig', 'name']
df_a = pd.read_csv(url_a, names=cols, sep='\s+', na_values='?').dropna()

# Using 'horsepower' (hp) to predict 'mpg'
X_p = df_a[['hp']].values
y_p = df_a['mpg'].values

# Transform to Degree 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_p)

xtr_p, xte_p, ytr_p, yte_p = train_test_split(X_poly, y_p, test_size=0.2, random_state=42)
model_p = LinearRegression().fit(xtr_p, ytr_p)

# --- Visualization ---
plt.figure(figsize=(10, 4))

# Linear Plot
plt.subplot(121); plt.scatter(yte, model.predict(xte)); plt.title("Linear: Actual vs Pred")

# Polynomial Plot
plt.subplot(122)
sort_idx = X_p.flatten().argsort()
plt.scatter(X_p, y_p, alpha=0.3)
plt.plot(X_p[sort_idx], model_p.predict(poly.transform(X_p[sort_idx])), color='red')
plt.title(f"Polynomial (deg 2) R2: {r2_score(yte_p, model_p.predict(xte_p)):.2f}")

plt.tight_layout(); plt.show()