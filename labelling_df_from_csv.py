#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:10:56 2019


ADDING LABELS FROM '/Users/thomasdrayton/Desktop/FYP/Labels_modified.csv'

    - ONLY PERFORMANCE CURVE : I HAVE MOST OF THE FAULTY ENGINE TESTS 2 OR 3 HAVENT BEEN INCLUDED BY FELIPE
    
    - NEED TO IMPORT df_subset as df_subset.spydata into workspace

@author: thomasdrayton
"""


import pandas as pd

#%% Import csv labels file
labels = pd.read_csv('/Users/thomasdrayton/Desktop/FYP/Labels_modified.csv')

#%% Removing rows from labels dataframe where nan exists   OR  import labels.spydata instead of running this section
idx = 0
for index,row in labels.iterrows():             # iterating through rows
    if(pd.isna(row['Step'])):                   # checking if Step column of row contains nan
        labels.drop(index= idx,inplace = True)  # drop that row
    idx+=1

#%% Adding a the Passed column: passed = 1,  Failed = 0
df_subset['Passed'] = 1    

#%% Adding O to specified windows failed tests: goes through rows of labels dataframe - for each row you have the engine code and step number - use those to find and enter values into Passed column in df_subset

for index,row in labels.iterrows():
    #print(row['CSV file'],row['Step'])
    df_subset.loc[(row['CSV file'],row['Step']),'Passed'] = 0
    
    
# =============================================================================
# #%% Adding 0 to all windows of failed tests
# 
# for index,row in labels.iterrows():
#     #print(row['CSV file'],row['Step'])
#     df_subset.loc[row['CSV file'],'Passed'] = 0
#     
# =============================================================================
#%% Modify insead of running from df_subset import
#df_subset.drop(columns = ['Passed'],inplace = True)
