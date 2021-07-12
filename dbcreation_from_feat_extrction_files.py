#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:54:54 2018

@author: thomasdrayton
"""

import pandas as pd
import os
import glob


indir ="/Users/thomasdrayton/Desktop/FYP/InformationExtraction"
os.chdir(indir)

fileList = glob.glob("*.csv")   # putting all the csv files into list


df_list = []

for file in fileList:
    df = pd.read_csv(file)
    df_list.append(df)
    


df_subset = pd.concat(df_list,axis=0)
df_subset.drop(df_subset.columns[0],axis=1, inplace = True)

df_subset.set_index(['eng test','window'], drop = True, inplace = True)

#df_subset = df_subset.transpose()


print(df_subset)
#%%
#df_n_IPvib = df_subset.drop(['tfuelv_f','toilv_f','tgtv_f','p30v_f'],axis=1)
# =============================================================================
# outfile = "/Users/thomasdrayton/Desktop/subset_ready_for_labelling.csv"
# df_subset.to_csv(outfile)
# =============================================================================


    
    