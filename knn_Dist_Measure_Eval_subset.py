#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 22:55:38 2018

Evaluating similarity Minkowski distance measures using kNN on subset database

    - database had incorrect labels on them due to lack of information provided by RR

@author: thomasdrayton
"""

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import pandas as pd
import os
import matplotlib.pyplot as plt

indir ="/Users/thomasdrayton/Desktop"
os.chdir(indir)

df = pd.read_csv('subset_database copy.csv')

X = df.iloc[:,range(2,8)]
y = df.iloc[:,8]
#print(X.head())
#print(type(X.iloc[0,]))

k_range = range(1,50,2)
accuracy = []
std_dev = []
# manhatten distance 
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, p=1)

    score = cross_val_score(knn, X, y, cv = 5, scoring = "accuracy")
    accuracy.append(score.mean())
    std_dev.append(score.std())
    

#plt.plot(std_dev, linewidth = 0.5, linestyle = '--',color = 'blue')
plt.plot(accuracy, linewidth = 0.5)
plt.title("Classification Accuracy for Minkowski Distance w/ kNN variation using 5 fold cross validation", fontsize = 7)
plt.tick_params(axis = 'both', labelsize = 4)
plt.xlabel('k',fontsize = 5)
plt.ylabel('mean fold score', fontsize =5)


# Euclidean
accuracy.clear()
std_dev.clear()
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, p=2)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    score = cross_val_score(knn, X, y, cv = 5, scoring = "accuracy")
    accuracy.append(score.mean())
    std_dev.append(score.std())

plt.plot(accuracy, linewidth = 0.5)
#plt.plot(std_dev, linewidth = 0.5, linestyle = '--',color = 'orange')

# Supremeum 
accuracy.clear()
std_dev.clear()
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k, p = float('inf'))
    score = cross_val_score(knn, X, y, cv = 5, scoring = "accuracy")
    accuracy.append(score.mean())
    std_dev.append(score.std())

plt.plot(accuracy, linewidth = 0.5)
#plt.plot(std_dev, linewidth = 0.5, linestyle = '--',color = 'green')

plt.legend(['Manhattan','Euclidean','Supremum'],loc = 4, fontsize = 7)

#png = 'kNN_Minkowski_comparison.png'
#plt.savefig("/Users/thomasdrayton/Desktop/FYP/Code/Figures/"+png,bbox_inches='tight',dpi = 1800) 

