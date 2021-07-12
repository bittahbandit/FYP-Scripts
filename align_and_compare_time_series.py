#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:29:02 2019

    - Align and compare raw data using a select subset from both classes
    
    - import IP_vib_n1v_step3_mixture
    
    - code uses imported spydata -> separates into faulty and normal
    -> finds peak n1v value and correponding time window
    -> plots performance curve starting just before peak for both faulty and normal
    
[Trying to see separation between faulty and non faulty by looking time series]

@author: thomasdrayton
"""


import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import more_itertools as mit


# variable to inspect for first 4 steps of performance curve
var = 'ps26v'



def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]




# change wkdir
indir ="/Users/thomasdrayton/Desktop/FYP/Code/All Normalised Data"
os.chdir(indir)


# create a list of normal and faulty 
filelist_n = []
filelist_f = []

for row in IP_vib_n1v_step3_mixture.iterrows():
    if(row[1][2]==0):
        filelist_f.append(row[0][0]+'.csv')
    else:
        filelist_n.append(row[0][0]+'.csv')


# intialise plot
fig,ax = plt.subplots(1,1,figsize =(16,16))

#filelist_n = 'T030035.csv'

# plot normal
for file in filelist_n:
    
    # read csv file
    df = pd.read_csv(file)
    
    # ====================================================================
    # calculate start of performance curve 
    # Remove sharp changes by taking the derivative
    df['dn1v'] = np.gradient(df['n1v'].rolling(center=True,window=50).mean())
    df['dn1v'] = df['dn1v']*10
    
    df_sep = df.drop(df[(df['dn1v'] < -0.001) | (df['dn1v'] > 0.001)].index, inplace=False)
    
    # create list of ranges that have consecutive values as tuples w/ ints
    l_withInts = list(find_ranges(df_sep.index))

    # removing int's to only have tuples
    l = []
    for i in l_withInts:     
        if(isinstance(i,tuple)):
            l.append(i)
            
    #removing small difference tuples
    cutoff = 80    # tuning to remove regions/tuples that dont have a time window greater than this value:  based on trying to get as big as possible (for the longer time series tests) to remove any captured windows that are transient  - but not so big (for smaller time series tests) as to remove any windows that are steady values      
    cut = []
    for i in l:
        diff = i[1] - i[0]
        if(diff<cutoff):
            cut.append(i)
            
    # remove all the elements of cut list that are in l list
    l = [x for x in l if x not in cut]
    
    
    # Find the highest speed in the data from the tuples from the actual data
    window_speed = []
    for i in l:
        initial_val = i[0]
        window_speed.append(df_sep.loc[initial_val,'n1v'])  # adding speed value at the the start of window to list
        
    highest_speed = max(window_speed)                  # seeing which is the highest in the list 

    # findig the window that corresponds to the peak n1v value
    idx = 0
    for i in l:
        if (df_sep.loc[i[0],'n1v'] == highest_speed):
            highest = i              # time window that corresponds to peak n1v
            break
        idx+=1          # idx that results when the loop breaks/exits correspons to the highest window
    
    
    # Only use windows: 1 before peak [but start at end] - to window 4    
    l = l[idx - 1:idx + 4]
    
    
    # ====================================================================
    # Plot for vibration comparison
    
    # plot n1v
    ax.plot(range(len(df.index[l[0][1]:l[-1][1]+1])),df.loc[l[0][1]:l[-1][1],var].values,c='black',linewidth=0.4,label='p')

    
#%%

# plot faulty 
for file in filelist_f:
    
    # read csv file
    df = pd.read_csv(file)
    
    # ====================================================================
    # calculate start of performance curve 
    # Remove sharp changes by taking the derivative
    df['dn1v'] = np.gradient(df['n1v'].rolling(center=True,window=50).mean())
    df['dn1v'] = df['dn1v']*10
    
    df_sep = df.drop(df[(df['dn1v'] < -0.001) | (df['dn1v'] > 0.001)].index, inplace=False)
    
    # create list of ranges that have consecutive values as tuples w/ ints
    l_withInts = list(find_ranges(df_sep.index))

    # removing int's to only have tuples
    l = []
    for i in l_withInts:     
        if(isinstance(i,tuple)):
            l.append(i)
            
    #removing small difference tuples
    cutoff = 80    # tuning to remove regions/tuples that dont have a time window greater than this value:  based on trying to get as big as possible (for the longer time series tests) to remove any captured windows that are transient  - but not so big (for smaller time series tests) as to remove any windows that are steady values      
    cut = []
    for i in l:
        diff = i[1] - i[0]
        if(diff<cutoff):
            cut.append(i)
            
    # remove all the elements of cut list that are in l list
    l = [x for x in l if x not in cut]
    
    
    # Find the highest speed in the data from the tuples from the actual data
    window_speed = []
    for i in l:
        initial_val = i[0]
        window_speed.append(df_sep.loc[initial_val,'n1v'])  # adding speed value at the the start of window to list
        
    highest_speed = max(window_speed)                  # seeing which is the highest in the list 

    # findig the window that corresponds to the peak n1v value
    idx = 0
    for i in l:
        if (df_sep.loc[i[0],'n1v'] == highest_speed):
            highest = i              # time window that corresponds to peak n1v
            break
        idx+=1          # idx that results when the loop breaks/exits correspons to the highest window
    
    
    # Only use windows: 1 before peak [but start at end] - to window 4        
    l = l[idx - 1:idx + 4]
    
    
    # ====================================================================
    # Plot for vibration comparison
    
    # plot n1v
    ax.plot(range(len(df.index[l[0][1]:l[-1][1]+1])),df.loc[l[0][1]:l[-1][1],var].values,c='red',linewidth=0.4,label='f')


#ax.legend(fontsize = 5)
ax.set_ylabel(var,fontsize = 8)
ax.set_xlabel('time',fontsize = 8)
ax.set_xlim([0,1700])
