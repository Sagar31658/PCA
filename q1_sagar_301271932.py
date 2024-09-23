# -*- coding: utf-8 -*-
"""Q1_Sagar_301271932.ipynb"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

# Load the dataset
data = arff.loadarff('F:\\Sem5\\Unsupervised\\Assignment 1\\mnist_784.arff')
df = pd.DataFrame(data[0])

# Convert 'class' column from byte strings to integers
for i in range(df.shape[0]):
  df.loc[i, 'class'] = int(df.loc[i, 'class'].decode('utf-8'))

df['class'] = df['class'].astype(float)

# Display sample images per class
for i in range(0, 10):
  showDF = df[df['class'] == i].sample(10)
  pixels = showDF.iloc[0, 1:].values.reshape(28, 28)
  plt.imshow(pixels)
  plt.show()

# PCA to reduce dimensionality to 2 components
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df.drop('class', axis=1)
y = df['class']
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_trf = pca.fit_transform(X_scaled)

# Plot the 2 principal components
plt.figure(figsize=(8, 6))
for label in y.unique():
    plt.scatter(X_trf[df['class'] == label, 0], X_trf[df['class'] == label, 1], edgecolor='k', label=f'{label}')
plt.title('PCA 2D Projection')
plt.legend()
plt.show()

# Reconstruct data using IncrementalPCA and compare images
from sklearn.decomposition import IncrementalPCA
pca2 = IncrementalPCA()
X_pca2 = pca2.fit_transform(X_scaled)
X_pca2_reconstructed = pca2.inverse_transform(X_pca2)

# Determine components for 95% variance
cumulative_variance = np.cumsum(pca2.explained_variance_ratio_)
num_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components for 95% variance: {num_components_95}")
