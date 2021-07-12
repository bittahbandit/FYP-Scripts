#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:53:27 2019

    - Create historgram of data 
    
    - Creates 4 subplots from the 4 columns selected in the variable cols
    
    - import df_subset.spydata [either all windows labelled or hand pick ones] 



@author: thomasdrayton
"""

import numpy as np
import matplotlib.pyplot as plt
import math


# Change to latex font ------------------------------------------------
plt.rcParams["font.family"] = "serif"


# cols
cols = ['n1v_f','tfuelv_f','toilv_f','LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
        'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
        'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
        'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

cols = ['tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

#X = df_subset.loc[:,cols].values
X = df_subset.loc[:,cols].values
y = df_subset.loc[:,"Passed"].values

# Just to get a rough idea how the data points of the two classes 
# visualize the distributions of 4 of the 26 features in 1-dimensional histograms.




#%%
# =============================================================================
# 
# label_dict = {0: 'Faulty', 1: 'Normal'}
# 
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))
# 
# for ax,cnt in zip(axes.ravel(), range(4)):  
#     
#     # set bin sizes
#     min_b = math.floor(np.min(X[:,cnt]))
#     max_b = math.ceil(np.max(X[:,cnt]))
#     bins = np.linspace(min_b, max_b, 25)
# 
#     # plottling the histograms
#     for lab,col in zip(range(0,2), ('blue', 'red')):
#         ax.hist(X[y==lab, cnt],
#                    color=col,
#                    label='%s' %label_dict[lab],
#                    bins=bins,
#                    alpha=0.5,)
#     ylims = ax.get_ylim()
#     
#     
#     # plot annotation
#     leg = ax.legend(loc='best', fancybox=True, fontsize=8)
#     leg.get_frame().set_alpha(0.5)
#     ax.set_ylim([0, max(ylims)+2])
#     ax.set_xlabel(cols[cnt])
# 
#     # hide axis ticks
#     ax.tick_params(axis="both", which="both", bottom="off", top="off",  
#             labelbottom="on", left="off", right="off", labelleft="on")
# 
#     # remove axis spines
#     ax.spines["top"].set_visible(False)  
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_visible(False)
#     ax.spines["left"].set_visible(False)    
# 
# axes[0][0].set_ylabel('count')
# axes[1][0].set_ylabel('count')
# =============================================================================


#%%

normal_fl = df_subset.loc[df_subset.Passed == 1]
faulty_fl = df_subset.loc[df_subset.Passed == 0]

normal_hl = df_subset000.loc[df_subset000.Passed == 1]
faulty_hl = df_subset000.loc[df_subset000.Passed == 0]
#%%

b_num = 130
fig, ax = plt.subplots(2,2,figsize=(16,10))
# Histogram IP vibration normal
min_b = np.min(normal_fl.IP_Vib_Magnitude_A_f.values)                                               # max of dimension
max_b = np.max(normal_fl.IP_Vib_Magnitude_A_f.values)                                               # min of dimension                                     # number of bins
bins = np.linspace(min_b, max_b, b_num)      
ax[0][0].hist(normal_fl.IP_Vib_Magnitude_A_f,label='Normal',bins=bins)                                    # plot normal data from dimension
ax[0][0].hist(faulty_fl.IP_Vib_Magnitude_A_f,label='Faulty',bins=bins,color='r')                          # plot faulty data from dimension
ax[0][0].set_xlabel('IP Vibration Magnitude',fontsize=16)
ax[0][0].set_ylabel('Frequency',fontsize=16)
ax[0][0].tick_params(axis='both',labelsize=13)

# Histogram IP vibration normal
min_b = np.min(normal_fl.toilv_f.values)                                               # max of dimension
max_b = np.max(normal_fl.toilv_f.values)                                               # min of dimension                                     # number of bins
bins = np.linspace(min_b, max_b, b_num)      
ax[1][0].hist(normal_fl.toilv_f,label='Normal',bins=bins)                                    # plot normal data from dimension
ax[1][0].hist(faulty_fl.toilv_f,label='Faulty',bins=bins,color='r')                          # plot faulty data from dimension
ax[1][0].set_xlabel('Oil Temperature',fontsize=16)
ax[1][0].set_ylabel('Frequency',fontsize=16)
ax[0][0].set_title('Fully-Labelled Performance Curve',fontsize=20)
ax[1][0].tick_params(axis='both',labelsize=13)


# Histogram IP vibration normal
min_b = np.min(normal_hl.IP_Vib_Magnitude_A_f.values)                                               # max of dimension
max_b = np.max(normal_hl.IP_Vib_Magnitude_A_f.values)                                               # min of dimension                                     # number of bins
bins = np.linspace(min_b, max_b, b_num)      
ax[0][1].hist(normal_hl.IP_Vib_Magnitude_A_f,label='Normal',bins=bins)                                    # plot normal data from dimension
ax[0][1].hist(faulty_hl.IP_Vib_Magnitude_A_f,label='Faulty',bins=bins,color='r')                          # plot faulty data from dimension
ax[0][1].set_xlabel('IP Vibration Magnitude',fontsize=16)
ax[0][1].set_ylabel('Frequency',fontsize=16)
ax[0][1].tick_params(axis='both',labelsize=13)

# Histogram IP vibration normal
min_b = np.min(normal_hl.toilv_f.values)                                               # max of dimension
max_b = np.max(normal_hl.toilv_f.values)                                               # min of dimension                                     # number of bins
bins = np.linspace(min_b, max_b, b_num)      
ax[1][1].hist(normal_hl.toilv_f,label='Normal',bins=bins)                                    # plot normal data from dimension
ax[1][1].hist(faulty_hl.toilv_f,label='Faulty',bins=bins,color='r')                          # plot faulty data from dimension
ax[1][1].set_xlabel('Oil Temperature',fontsize=15)
ax[1][1].set_ylabel('Frequency',fontsize=15)
ax[0][1].set_title('Hand-Labelled Performance Curve',fontsize=20)
ax[1][1].tick_params(axis='both',labelsize=13)

#fig.text(0.06, 0.5, 'Frequency', va='center', rotation=90,fontsize=16)          # legend



# Put a legend below current axis
ax[1][0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.09),
          fancybox=True,
          ncol=2,
          fontsize=16)

#ax.legend()

