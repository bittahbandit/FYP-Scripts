#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:09:05 2019

    - gmm exmpale taken from

https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html


@author: thomasdrayton
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse


# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=[40,1300], centers=[(2,3),(0.3,2)],
                       cluster_std=0.40, random_state=0)
#X = X[:, ::-1] # flip axes for better plotting

# ------------------------------- plot raw data ------------------------------
fig,ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], s=10, cmap='viridis')


print(len(y_true[y_true == 1]))

#%%
# ------------------------- predict data using gmm ---------------------------
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
fig,ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis');




# ------------------ Accessing predicted probabilities -----------------------
#  returns a matrix of size [n_samples, n_clusters]
probs = gmm.predict_proba(X)
print(probs[:5].round(3))




# ----------- Visualize uncertainty as proportional to the size ---------------
size = 30 * probs.max(1) ** 5  # square emphasizes differences
fig,ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size);



# ----------- Functions for plotting gmm ---------------

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)