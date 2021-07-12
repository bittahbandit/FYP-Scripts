#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:46:08 2018

@author: thomasdrayton
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from matplotlib import rc
import numpy as np
import more_itertools as mit


#import random
#from sklearn import preprocessing, cross_validation, neighbors



indir ="/Users/thomasdrayton/Desktop/FYP/Code/All Normalised Data"
os.chdir(indir)



fileList = glob.glob("*.csv")   # putting all the csv files into list
file = 'T001853.csv'

df = pd.read_csv(file)

# Database creation
selection = ['eng test','window','n1v_f','tfuelv_f','toilv_f',
             'LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
             'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
             'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
             'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']
db = pd.DataFrame(columns=selection, dtype = float)

# retaining orig. for n1v later 
n1v_orig = df[['n1v']]
#print(n1v.head())

# Variable to analyse REMOVE F
var = 'IP_Vib_Magnitude_A'
var2 = 'tfuelv'


# Remove sharp changes by taking the derivative
df['dn1v'] = np.gradient(df['n1v'].rolling(center=True,window=50).mean())       # 50
df['dn1v'] = df['dn1v']*10                                                      # 10





# =============================================================================
# #plot gradient for determining separation
# fig2, ax2 = plt.subplots(2, sharex=True, figsize=(10,6))
# ax2[0].plot(df.index,'n1v',data=df,linewidth=0.9 ,c='k')
# ax2[1].plot(df.index,'dn1v',data=df,linewidth=0.9, c='k')
# 
# ax2[1].tick_params(axis = 'both',labelsize = 9)
# ax2[0].tick_params(axis = 'both',labelsize = 9)
# 
# fig2.text(0.94, 0.70, r'$f_{n1v}(t)$', ha='center', fontsize = 13)
# fig2.text(0.95, 0.25, r'$\frac{d}{dt}(f_{n1v}(t))$', ha='center', fontsize = 13)
# 
# fig2.text(0.5, 0.04, 'Time (s)', ha='center',fontsize=12)
# fig2.text(0.04, 0.5, 'Signal Magnitude', va='center', rotation='vertical',fontsize=12)
# fig2.text(0.52, 0.92, 'Time Series Segmentation', ha='center',fontsize=14)
# =============================================================================






# looking at variance of dn1v in selected window
# =============================================================================# displaying gradient in window 
#n = 3490
#r = range(n,n+60)

#print(df.loc[r,'dn1v'])
# =============================================================================






# Change to latex font ------------------------------------------------
plt.rcParams["font.family"] = "serif"

# Plot separation regions ---------------------------------------
fig, ax = plt.subplots(2,1,figsize=(10,6),sharex=True)

ax[0].plot(df.index,'n1v',data = df,linewidth=0.3,c='k')
ax[0].plot(df.index,var,data = df,linewidth=0.3,label=var,c='red')

ax[1].plot(df.index,'n1v',data = df,linewidth=0.3,c='k')
ax[1].plot(df.index,var2,data = df,linewidth=0.3,label=var,c='red')


# Plot specified variable 



fig.text(0.5, 0.04, 'Time (s)', ha='center',fontsize=12)
fig.text(0.04, 0.5, 'Signal Magnitude', va='center', rotation='vertical',fontsize=12)

fig.text(0.91, 0.72, 'IP Vibration Magnitude', va='center', rotation=-90,fontsize=11,color='red')
fig.text(0.91, 0.27, 'Fuel Temperature', va='center', rotation=-90,fontsize=11,color='red')



# Remove transients
df.drop(df[(df['dn1v'] < -0.001) | (df['dn1v'] > 0.001)].index, inplace=True)

#ax.set_title(''+file.replace('.csv',''),fontsize = 11)
fig.text(0.52, 0.92, 'Mean Value Extraction', ha='center',fontsize=14)








# =============================================================================
# Create window


#[list(group) for group in mit.consecutive_groups(df.index)]

def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]

# create list of ranges that have consecutive values as tuples w/ ints
l_withInts = list(find_ranges(df.index))
#print(isinstance(l[0],tuple))

# removing int's to only have tuples
l = []
for i in l_withInts:       # check this loop
    #print(i)
    if(isinstance(i,tuple)):
        #print("int:",i)
        #l.remove(i)
        l.append(i)
#print(l)
      
# =============================================================================
# count = 0
# # filtering to obtain large differences
# for tup in range(len(l)):
#     diff = l[tup][1] - l[tup][0]
#     #print(diff)
#     #print(type(l[tup][0]),',',l[tup][1])
#     if(diff<100):
#         del l[0]
#     count+=1
# =============================================================================
#removing small difference tuples
# NEED TO FIX:  FIND WAY AROUND MISSING ITERATIONS **************

#print('='*30)
#print(l)
cutoff = 80    # tuning to remove regions/tuples that dont have a time window greater than this value:  based on trying to get as big as possible (for the longer time series tests) to remove any captured windows that are transient  - but not so big (for smaller time series tests) as to remove any windows that are steady values      
#count = 0
cut = []
for i in l:
    #print(i)
    diff = i[1] - i[0]
    #print(i)
    if(diff<cutoff):
        #print(i,"remove")
        cut.append(i)
        #del l[count]

# remove all the elements of cut list that are in l list
l = [x for x in l if x not in cut]
#print("l is :",l)

#print(lcut)
print('number of windows: ', len(l))

# =============================================================================
# # Gauging length of of time step is held constant for
# diff_list = []
# for i in l:
#     diff = i[1] - i[0]
#     diff_list.append(diff)
# =============================================================================



# Find the highest speed in the data from the tuples from the actual data
window_speed = []
for i in l:
    initial_val = i[0]
    window_speed.append(df.loc[initial_val,'n1v'])  # adding speed value at the the start of window to list
    
highest_speed = max(window_speed)                  # seeing which is the highest in the list 

# findig the window that corresponds to the peak n1v value
idx = 0
for i in l:
    if (df.loc[i[0],'n1v'] == highest_speed):
        highest = i             # not used later on: check to check if it's right
        break
    idx+=1
print('starting pt is in window',idx)

# Keep the window previous to the peak window for time_constant calc
ptpw = l[idx - 1]

# Only keep 5 steps including the top one
l = l[idx:idx + 7]

#plotting final performance curve regions
for i in l:
    ax[0].scatter(range(i[0],i[1]+1),df.loc[i[0]:i[1],'n1v'],data=df,s=2,
               c='darkblue')
    ax[1].scatter(range(i[0],i[1]+1),df.loc[i[0]:i[1],'n1v'],data=df,s=2,
               c='darkblue')

# =============================================================================
# # Plotting points to show step separation after refinement
# startpt = []
# ones = []
# for i in l:
#     startpt.append(i[0])
#     ones.append(1)
# 
# fig, ax = plt.subplots(1)
# ax.scatter(n1v_orig.index,'n1v',data = n1v_orig,s=0.12)
# ax.scatter(startpt,ones,s=10)
# ax.legend(['n1v','n1v sep'])
# ax.set_title('Step separation after refinement')
# =============================================================================

#Checking if the starting vlaue is above 0.93, then dont include it in subset
print('Is the 1st window n1v value greater than 0.9? ',df.loc[l[0][0],'n1v']>0.9)
print('It is ',df.loc[l[0][0],'n1v'])

# =============================================================================
# for current file / engine code: in window 1 find mean value
#selection = ['n1v_f','tfuelv_f','toilv_f','HP_Vib_Magnitude_A_f','tgtv_f','p30v_f']

#add windows to columns of db dataframe and initialising them with NaN 
#for i in range(len(l)):
    #db[i] = np.nan
    



count = 0                  
# Iterating through each window w
for w in range(len(l)):
    
    win = l[w] #window tuple 
    win_range = range(win[0],win[1]) # creating range from window tuple
    
    mid = int(win[1] - (win[1]-win[0])/2)                    #find middle index value

    #n_featrue
    n1v_f = df.loc[win_range,'n1v'].mean()
    #print(n_f)
    db.loc[w,'n1v_f'] = n1v_f

    # tfuelv    RECODE this so it's not buggy
    tfuelv = df.loc[win_range,'tfuelv']   #steady state value in the window
    #tfuelv_f = (tfuelv.iloc[-1] - tfuelv.iloc[0])/len(win) # slope 
    tfuelv = tfuelv.rolling(cutoff,center=True).mean()
    tfuelv_f = tfuelv[mid]
    #for i in tfuelv:
        #if(~np.isnan(i)):
            #tfuelv_f = i                                     
    db.loc[w,'tfuelv_f'] = tfuelv_f


    # toilv                         RECODE this so it's not buggy
    toilv = df.loc[win_range,'toilv']          # smooth and output final value  
    toilv = toilv.rolling(cutoff,center=True).mean()
    toilv_f = toilv[mid]
    db.loc[w,'toilv_f'] = toilv_f
    
    # HP_vibration                     
    HP_Vib_Magnitude_A_f= df.loc[win_range,'HP_Vib_Magnitude_A'].mean()      # finding middle in averaged series and assgining it to the feature  
    db.loc[w,'HP_Vib_Magnitude_A_f'] = HP_Vib_Magnitude_A_f        # place feature into database
    
    # IP_Vib_Magnitude_A
    IP_Vib_Magnitude_A_f = df.loc[win_range,'IP_Vib_Magnitude_A'].mean()            
    db.loc[w,'IP_Vib_Magnitude_A_f'] = IP_Vib_Magnitude_A_f
    
    #plot to check vibration feature
    #fig, ax = plt.subplots(1)
    #ax.plot(df.index,'HP_Vib_Magnitude_A',data = df,linewidth=0.12)
    #ax.plot(df.index,'n1v',data = df,linewidth=0.12)
    #mid = int(win[1] - (win[1]-win[0])/2)                    #find middle index value
    #ax.scatter(mid,HP_Vib_Magnitude_A_f,s = 10,c = 'r')        # PLOT IP_Vib FEATURE EXTRACTION DOT 
    #ax.set_title('T000103')
    #ax.legend([])
    #fig.savefig("/Users/thomasdrayton/Desktop/toilv_smoothed.png",bbox_inches='tight',dpi = 1800)
    
    # tgtv
    tgtv_f = df.loc[win_range,'tgtv'].mean()              # mean in given window
    db.loc[w,'tgtv_f'] = tgtv_f
    
    # t20v
    t20v_f = df.loc[win_range,'t20v'].mean()     # mean 
    db.loc[w,'t20v_f'] = t20v_f
    
    # t25v
    t25v_f = df.loc[win_range,'t25v'].mean()     
    db.loc[w,'t25v_f'] = t25v_f
    
    # t30v
    t30v_f = df.loc[win_range,'t30v'].mean()     
    db.loc[w,'t30v_f'] = t30v_f
    
    # ps26v
    ps26v_f = df.loc[win_range,'ps26v'].mean()     # mean 
    db.loc[w,'ps26v_f'] = ps26v_f
    
    # p30v
    p20v_f = df.loc[win_range,'p20v'].mean()     # mean 
    db.loc[w,'p20v_f'] = p20v_f
    
    # P42
    P42_f = df.loc[win_range,'P42'].mean()     # mean 
    db.loc[w,'P42_f'] = P42_f
    
    # P44
    P44_f = df.loc[win_range,'P44'].mean()  
    db.loc[w,'P44_f'] = P44_f
    
    # tcarv_S
    tcarv_S_f = df.loc[win_range,'tcarv_S'].mean()  
    db.loc[w,'tcarv_S_f'] = tcarv_S_f
    
    
    # tcafv_S
    tcafv_S_f = df.loc[win_range,'tcafv_S'].mean()  
    db.loc[w,'tcafv_S_f'] = tcafv_S_f
    
    # tprtrimmed
    tprtrimmed_f = df.loc[win_range,'tprtrimmed'].mean()  
    db.loc[w,'tprtrimmed_f'] = tprtrimmed_f
    
    # trav
    trav_f = df.loc[win_range,'trav'].mean()  
    db.loc[w,'trav_f'] = trav_f
    
    # poilv_S
    poilv_S_f = df.loc[win_range,'poilv_S'].mean()  
    db.loc[w,'poilv_S_f'] = poilv_S_f
    
    # tsasv_S
    tsasv_S_f = df.loc[win_range,'tsasv_S'].mean()  
    db.loc[w,'tsasv_S_f'] = tsasv_S_f
    
    # adding engine test code
    db.loc[w,'eng test'] = file.replace(".csv","")
    
    # adding window
    db.loc[w,'window'] = count
    count+=1
    
    
    
    
        
        
        

    
    
    # Plot features
    ax[0].scatter(mid,IP_Vib_Magnitude_A_f,marker='x',c='darkblue')
    ax[1].scatter(mid,tfuelv_f,marker='x',c='darkblue')





# mark faulty windows
f_win = []
for idx_i,i in enumerate(l):
    for ii in f_win:
        if(ii==idx_i):
            ax.scatter(range(i[0],i[1]+1),df.loc[i[0]:i[1],'n1v'],data=df,s=6,
                       c='red')


# Legend --------------------------------------------------------------
# Shrink current axis's height by 10% on the bottom
box = ax[0].get_position()
ax[0].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# legend
black_patch = mpatches.Patch(color='black', label='Rotational Shaft Speed: n1v')
darkblue_patch = mpatches.Patch(color='darkblue', label='Performance Curve Segmentation')
red_patch = mpatches.Patch(color='red', label='Measured Variable')
#red_patch = mpatches.Patch(color='red', label='Extracted Mean Value')
darkblue_cross = plt.scatter([],[],marker='x',s=30,c='darkblue',label='Extracted Mean Value')


ax[0].legend(handles=[black_patch, darkblue_patch, red_patch, darkblue_cross],
          loc='upper center',
          bbox_to_anchor=(0.5, -0.09),
          fancybox=True,
          ncol=4,
          fontsize = 9)




# =============================================================================
# Database creation
#outfile = "/Users/thomasdrayton/Desktop/FeatureExtraction/T000103.csv"



# =============================================================================
# eng_code=[]
# for i in range(len(db.columns)):
#     #print(i)
#     eng_code.append(file.replace(".csv",""))
# 
# windows = db.columns
# db.columns = [eng_code,windows]
# =============================================================================


#print(db)
#db.to_csv(outfile)

#print(db.loc[["HP_Vib_Magnitude_A_f"]])

# =============================================================================
# #Validate by plotting both
# 
# end_window = []
# for i in l:
#     end_window.append(i[1])
# 
# fig, ax = plt.subplots(1)
# ax.plot(df.index,'HP_Vib_Magnitude_A',data = df,linewidth=0.2)
# ax.plot(df.index,'HP_Vib_Magnitude_A',data = df,linewidth=0.2)
# ax.scatter(end_window,db.loc[["HP_Vib_Magnitude_A_f"]], c = 'red', s = 5)
# ax.legend(["HP_Vib_Magnitude_A","HP_Vib_Magnitude_A_f"])
# ax.set_title("T000103 Feature Selection verification")
# fig.savefig("/Users/thomasdrayton/Desktop/T000103 Feature Selection verification.png",bbox_inches='tight',dpi = 1800)
# =============================================================================

#%%

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#S_t = df.loc[i[0],'tcarv_S'] - 0.632*np.abs(df.loc[i[0],'tcarv_S'] - df.loc[i[1],'tcarv_S'])
#T = df.loc[i[0]:i[1],'tcarv_S'] 
#T = min(range(len(T.values)), key=lambda i_c: abs(T.values[i_c]-S_t))
#print(df.loc[win[0]:win[1],'tcarv_S'])
#plt.plot(df.loc[win[0]:win[1],'tcarv_S'].index,df.loc[win[0]:win[1],'tcarv_S'].rolling(center=True,window=25).mean()) 

#%%

def time_constant(param,ci,ptpw):
    '''
    ci --->  is current iteration  i.e. w 
    ptpw ---> is previous to peak window 
    param ---> is the parameter you want the time constant to be retrieved from
    
    Calculates time constant T depending on if rising or falling exponential
    '''
    
    # 0.63 of signal is time constant [signal percent]
    sp = 1 - np.exp(-1)
    
    # previous window excluding first window
    prev_win = l[ci - 1]
    
    # current window
    c_win = l[ci]
    
    if(ci==0 and (df.loc[ptpw[1],param] < df.loc[c_win[1],param])): # if it's the first and rising
        print('here1')
        # signal value at time constant
        S_t = df.loc[ptpw[1],param] + sp * np.abs(df.loc[ptpw[1],param] - df.loc[win[1],param])
        
        # time constant - smoothed param values to migitate against signal noise
        param_vals = df.loc[ptpw[1]:win[1],param].rolling(center=True,window=25).mean()  #smooth a degree
        param_vals.fillna(0,inplace = True)
        
        T = min(range(len(param_vals.values)), key=lambda i_c: abs(param_vals.values[i_c]-S_t))
    
    elif(ci==0 and (df.loc[ptpw[1],param] > df.loc[c_win[1],param])): #if it's the first and falling 
        print('here2')
        
        # signal value at time constant
        S_t = df.loc[ptpw[1],param] - sp * np.abs(df.loc[ptpw[1],param] - df.loc[win[1],param])
        
        # time constant - smoothed param values to migitate against signal noise
        param_vals = df.loc[ptpw[1]:win[1],param].rolling(center=True,window=25).mean()  #smooth a degree
        param_vals.fillna(0,inplace = True)

        T = min(range(len(param_vals.values)), key=lambda i_c: abs(param_vals.values[i_c]-S_t))
    
    
    elif(df.loc[c_win[0],param] > df.loc[c_win[1],param]):    # if start signal is greater than end -> decay(falling)
        print('here3')
              
        # Signal value at T = starting signal - 0.63(signal differnce)
        S_t = df.loc[prev_win[1],param] - sp * np.abs(df.loc[prev_win[1],param] - df.loc[win[1],param])

        # time constant - you could smoothed param values to migitate against signal noise
        param_vals = df.loc[prev_win[1]:win[1],param].rolling(center=True,window=25).mean()  #smooth a degree                                              # end of previous window
        param_vals.fillna(0,inplace = True)

        T = min(range(len(param_vals.values)), key=lambda i_c: abs(param_vals.values[i_c]-S_t))
        
    else: # if it's not the first and rising - don't think one exists
        print('here4')

        # Signal value at T = starting signal - 0.63(signal differnce)
        S_t = df.loc[prev_win[1],param] + sp * np.abs(df.loc[prev_win[1],param] - df.loc[win[1],param])

        # time constant - smoothed param values to migitate against signal noise
        param_vals = df.loc[prev_win[1]:win[1],param].rolling(center=True,window=25).mean()  #smooth a degree                                              # end of previous window
        param_vals.fillna(0,inplace = True)

        T = min(range(len(param_vals.values)), key=lambda i_c: abs(param_vals.values[i_c]-S_t))
        
    return T,param_vals



T, param_vals = time_constant('tcarv_S',0,ptpw)
#%%
param_vals = df.loc[l[w - 1][1]:win[1],'tcarv_S'].rolling(center=True,window=25).mean()  #smooth a degree                                              # end of previous window
param_vals.fillna(0,inplace = True)

print((df.loc[l[0 - 1][1],'tcarv_S'] < df.loc[l[1],'tcarv_S']))


