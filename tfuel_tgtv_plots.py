#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:20:39 2019

    - Plots any variable from imported feature database 

    - IMPORT df_subset BEFORE
    
    - Run section by section  &  read section before you run it.

@author: thomasdrayton
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Change to the variables that you want to plot
x_param = 'tgtv_f'
y_param = 'tfuelv_f'


#%% Plot each step in the performance curve - color coded by step

# =============================================================================
# #Create list where each element contains a dataframe where all the indices are from a sinlge window
# windows_df_list = []
# for w in range(6):
#     windows_df_list.append(df_subset.loc(axis=0)[:,w])  # - indexes all the engine tests, of each window
#     
# 
# fig,ax = plt.subplots()
# 
# idx = 0
# colours = ['k','r','royalblue','darksalmon','mediumseagreen','orange']
# for i,colour in zip(windows_df_list,colours):
#     ax.scatter(i.loc[:,x_param],i.loc[:,y_param],c = colour,s = 12)
# =============================================================================

#%%  normal and abnormal datasets
normal = df_subset.loc[df_subset['Passed'] == 1]
abnormal = df_subset.loc[df_subset['Passed'] == 0] 

#%%  1D Array for x (n1v_f) and y (IP_vib) 
x = normal[x_param].values
y = normal[y_param].values

fig,ax = plt.subplots(1,1,figsize=(16,16))

ax.scatter(x,y,label = "Normal",s=1)
ax.scatter(abnormal[x_param].values,abnormal[y_param].values,c = 'r',
           label='Abnormal',s=35,marker = 'x',linewidth = 0.7)

fig.text(0.5, 0.04, x_param, ha='center')
fig.text(0.04, 0.5, y_param, va='center', rotation='vertical')

ax.legend()