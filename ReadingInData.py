#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 23:00:27 2018

Practice reading in data

@author: thomasdrayton
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# moving to normalised data for wkdir 
indir ="/Users/thomasdrayton/Desktop/FYP/Code/Normalised Data"
os.chdir(indir)

dfList = []
df = pd.read_csv('T000103.csv')
dfList.append(df['n2v'])

print(df.head())
print(df["n1v"])





#dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}

#print(dataset)
'''
# Finding the dimensionality of your dataframe
shp = df.shape
print(shp)
# Column headers and rows are not included in the dimensions

# Checking which class your object belongs
typ = type(shp)
print(typ)


# Selecting any row
r = df['id']
print(r)


# Selecting only including certain data in your selection
onlyMitosisIs1 = df[df.mitosis == 1]
print(onlyMitosisIs1)

# Selecting only specific features/columns 
firstAndLast = df[['id','class']]
moreSelected = df[['id','mitosis','class']]
print(moreSelected)
# double square brackets because you are selecting mulitple features


# Selecting specific data value: rw 23 and column 'class'
df_point = df.loc[23,'class']
print(df_point)

# useful to replace the row indexing with meaningful name instead of integers
df.set_index('id',inplace=True)
print(df.head())

# retrieving statistical data from your data frame
print(df.describe())

# Adding a new column to data
df['new feature'] = df['id']
print(df.head())
'''



'''
# Using a numpy array instead of a list because it uses less memory & faster
X = np.array(df)

# Indexing array
# -------------
# Index first row
print(X[0])

# Index first column
print(X[:,0])
'''



