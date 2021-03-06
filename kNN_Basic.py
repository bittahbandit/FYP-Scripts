#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:12:19 2018

@author: thomasdrayton
"""

import os
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')


# moving to normalised data for wkdir 
indir ="/Users/thomasdrayton/Desktop/FYP/Code/Data Separated into Varibles"
os.chdir(indir)



# creating 2 classes and their features
dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [4000,0.5]

for group in dataset:
    print("group:",group)


def k_nearest_neighbours(data,predict,k=3):
    if len(data)>= k:
        warnings.warn('K is set to a value less thn total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    
    return vote_result


#result = k_nearest_neighbours(dataset, new_features,k=3)
#print(result)



'''
[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],s=100)
plt.show()

[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],s=100,color=result)
plt.show()'''