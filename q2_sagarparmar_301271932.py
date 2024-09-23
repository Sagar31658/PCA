# -*- coding: utf-8 -*-
"""Q2_SagarParmar_301271932.ipynb"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

import warnings
warnings.filterwarnings('ignore')

# Generate the swiss roll dataset
from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1300, random_state=32)

# 3D scatter plot of the swiss roll
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='viridis', marker='o', alpha=0.8)
ax.set_title('3D Scatter Plot of Swiss Roll')
plt.show()

# Kernel PCA with different kernels
from sklearn.decomposition import KernelPCA

kpca_linear = KernelPCA(n_components=2, kernel='linear')
kpca_rbf = KernelPCA(n_components=2, kernel='rbf')
kpca_sigmoid = KernelPCA(n_components=2, kernel='sigmoid')

linear = kpca_linear.fit_transform(X)
rbf = kpca_rbf.fit_transform(X)
sigmoid = kpca_sigmoid.fit_transform(X)

# Plotting the KPCA results for each kernel
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Linear Kernel PCA
axs[0].scatter(linear[:, 0], linear[:, 1], c=t, cmap='viridis', alpha=0.8)
axs[0].set_title('KPCA with Linear Kernel')

# RBF Kernel PCA
axs[1].scatter(rbf[:, 0], rbf[:, 1], c=t, cmap='viridis', alpha=0.8)
axs[1].set_title('KPCA with RBF Kernel')

# Sigmoid Kernel PCA
axs[2].scatter(sigmoid[:, 0], sigmoid[:, 1], c=t, cmap='viridis', alpha=0.8)
axs[2].set_title('KPCA with Sigmoid Kernel')

plt.show()

# Binary classification based on the median of 't'
from sklearn.model_selection import train_test_split
y = (t > np.median(t)).astype(int)  # Creating binary labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with GridSearchCV for KernelPCA parameters
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('kpca', KernelPCA(n_components=2, kernel='sigmoid')),
    ('log_reg', LogisticRegression())
])

# Grid search over KPCA kernels and gamma values
param_grid = {
    'kpca__kernel': ['rbf', 'sigmoid', 'poly', 'linear'],
    'kpca__gamma': np.logspace(-4, 1, 6),
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Display the best parameters and scores
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))

# Access the grid search results
results = grid_search.cv_results_
