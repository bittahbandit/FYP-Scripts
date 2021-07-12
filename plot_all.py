#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:20:39 2019

    - Plots all non repeated combinations of variables from imported feature database 

    - IMPORT df_subset BEFORE 
    
    - Run section by section  &  read section before you run it.

@author: thomasdrayton
"""

import numpy as np
import matplotlib.pyplot as plt


features = ['n1v_f','tfuelv_f','toilv_f', 'LP_Vib_Magnitude_A_f',
            'IP_Vib_Magnitude_A_f','HP_Vib_Magnitude_A_f','t20v_f','t25v_f',
            't30v_f','p20v_f','ps26v_f','p30v_f','P42_f','P44_f','tgtv_f',
            'tcarv_S_f','tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f',
            'tsasv_S_f']


done = []
count = 0
#  Change to the variables that you want to plot
for i in features:
    for j in features:
        x_param = i
        y_param = j
        
        if(i==j):
            break
        if([i,j] in done):
            break
        if([j,i] in done):
            break

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
        
       
        fig,ax = plt.subplots(1,1,figsize=(16,10))
        
        ax.scatter(x,y,label = "Normal",s=1)
        ax.scatter(abnormal[x_param].values,abnormal[y_param].values,c = 'r',
                   label='Abnormal',s=35,marker = 'x',linewidth = 0.7)
        
        fig.text(0.5, 0.04, x_param, ha='center')
        fig.text(0.04, 0.5, y_param, va='center', rotation='vertical')
        
        ax.legend()
        
        png = "{0} - {1}.png".format(i,j)
        fig.savefig("/Users/thomasdrayton/Desktop/FYP/Figures/feature space & subspaces/All variables with every window labelled in faulty test/"+png,bbox_inches='tight',dpi = 500) 
        
        plt.close(fig)        
        
        done.append([i,j])
        
        count+=1
        print(count)
    
    
#%%
