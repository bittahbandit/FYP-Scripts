#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:37:19 2019

    - understanding one class svm from sklearn tutorial

@author: thomasdrayton
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

# create a meshgrid for plot between 5 and -5 for both axis
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))

# Generate train data  -  all from the same class: white dots
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# plot traing data 
fig,ax = plt.subplots()
ax.scatter(X_train[:, 0], X_train[:, 1], c='white', s=40, edgecolors='k')
ax.set_title("training data")

# Generate some regular novel observations - test data from the same class
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

fig,ax = plt.subplots()
ax.scatter(X_test[:, 0], X_test[:, 1], c='purple', s=40, edgecolors='k')
ax.set_title("test data")

# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

fig,ax = plt.subplots()
ax.scatter(X_outliers[:, 0], X_outliers[:, 1], c='yellow', s=40, edgecolors='k')
ax.set_title("abnormal data")

# -------------------------------- Training -----------------------------------

# Specify the model and instantiate parameters
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

# fit the model to the training data [figure 1]
clf.fit(X_train)

# Using trained model to predict output of training predictions
y_pred_train = clf.predict(X_train)

# plot train results
count = 0  # count number of abnormal
fig,ax = plt.subplots()
for train,out in zip(X_train,y_pred_train):
    #print(train,out)
    if(out==-1):
        ax.scatter(train[0],train[1], c='red', s=40, edgecolors='k')
        count+=1
    else:
        ax.scatter(train[0],train[1], c='white', s=40, edgecolors='k')
ax.set_title("Results of training: white=normal, red=??")
print('Number of training abnormal: ',count,'/',len(y_pred_train))


# -------------------------------- Testing -----------------------------------

# Use the same model from training data, to predict outcome of test data
y_pred_test = clf.predict(X_test)

# plot test results
count = 0  # count number of abnormal
fig,ax = plt.subplots()
for test,out in zip(X_test,y_pred_test):
    #print(train,out)
    if(out==-1):
        ax.scatter(test[0],test[1], c='red', s=40, edgecolors='k')
        count+=1
    else:
        ax.scatter(test[0],test[1], c='white', s=40, edgecolors='k')
ax.set_title("Results of training: white=normal, red=??")
print('Number of testing abnormal: ',count,'/',len(y_pred_test))


# -------------------------------- Outlier Test -------------------------------

# Use the same model from training data, to predict outcome of outlier data
y_pred_outliers = clf.predict(X_outliers)

# plot outlier results
count = 0  # count number of abnormal
fig,ax = plt.subplots()
for outliers,out in zip(X_outliers,y_pred_outliers):
    #print(train,out)
    if(out==-1):
        ax.scatter(outliers[0],outliers[1], c='red', s=40, edgecolors='k')
        count+=1
    else:
        ax.scatter(outliers[0],outliers[1], c='white', s=40, edgecolors='k')
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_title("Results of outliers: white=normal, red=??")
print('Number of outliers abnormal: ',count,'/',len(y_pred_outliers))


# Obtaining the number of data points classes as abnormal from training,testing and outlier test (i've already done this by printing results)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


# ------------------------------- Plot from tutorial --------------------------
# plot the line, the points, and the nearest vectors to the plane

# initialise decision fucntion using meshgrid results
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  
Z = Z.reshape(xx.shape)  # change from 1D array to shape dimensions as meshgrid (500,500)


fig,ax = plt.subplots()
ax.set_title("Novelty Detection")
ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
ax.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = ax.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = ax.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,edgecolors='k')
c = ax.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,edgecolors='k')
ax.axis('tight')
ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))
ax.legend([a.collections[0], b1, b2, c],["learned frontier", "training observations","new regular observations", "new abnormal observations"],loc="upper left", prop=matplotlib.font_manager.FontProperties(size=11))
ax.set_xlabel("error train: %d/200 ; errors novel regular: %d/40 ; errors novel abnormal: %d/40% (n_error_train, n_error_test, n_error_outliers)")