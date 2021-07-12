#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:01:24 2019

    - Gaussian Process Regression [Modelling Normality]
    
    - Need to import IMPORT df_subset_p_curve_labelled BEFORE

@author: thomasdrayton
"""


import numpy as np
import matplotlib.pylab as plt
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


#%% Preparing IP vib n1v data 

# Remove all unneeded columns
#df_subset.drop(labels =['tfuelv_f','toilv_f','tgtv_f','p30v_f'],axis = 1, inplace = True)

# normal and abnormal datasets
normal = df_subset.loc[df_subset['Passed'] == 1]
abnormal = df_subset.loc[df_subset['Passed'] == 0] 

# 1D Array for x (n1v_f) and y (IP_vib) 
x = normal['n1v_f'].values
y = normal['IP_Vib_Magnitude_A_f'].values

# Reorganise x and y values for normal data so that it ascends
# Create list of tuples of xy
xy = []
for i in range(len(x)):
    xy.append((x[i],y[i]))
    
# Sort list so that ascends by first element in tuple
xy = sorted(xy,key=lambda x : x[0])

# Put them back into individual lists
x= []
y = []
for i in xy:
    x.append(i[0])
    y.append(i[1])
    
#%% Convert to a numpy array
x = np.array(x)
y = np.array(y)
#fig,ax = plt.subplots(1,1,figsize = (16,16))
#ax.scatter(x,y,label = "Normal",s=1)
    
#%% GP regression


# Change to latex font ------------------------------------------------
plt.rcParams["font.family"] = "serif"
        
fig,ax = plt.subplots(2,2,figsize = (16,16))

fig.text(0.52, 0.92, 'Effects of Matérn Covariance Function Hyperparameters', ha='center',fontsize=14)

for i,l in enumerate([0.05,0.1]): # lengthscale
    for j,nu in enumerate([3/2, 5/2]): # smoothness
        
        #        
        kernel = ConstantKernel() + Matern(length_scale=l, nu=nu) + WhiteKernel(noise_level=0.037)
        #kernel  = Matern(length_scale=l, nu=nu) + WhiteKernel(noise_level=1)

        X = x.reshape(-1, 1)


        # instantiate a GaussianProcessRegressor object w/ the custom kernel, and call 
        # its fit method, passing the input (X) and output (y) arrays
        
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, optimizer=None)  # remove optimizer parameter for optimization
        gp.fit(X, y)
        
        print(i,j)
        print(gp.kernel_)
        
        #% Not sure about this
        x_pred = np.linspace(0, 1,900).reshape(-1,1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        
        
        ax[i][j].scatter(x,y,label = "Normal",s=1)
        ax[i][j].plot(x_pred, y_pred, color='grey', label='Prediction')
        ax[i][j].fill(np.concatenate([x_pred, x_pred[::-1]]),
                np.concatenate([y_pred - 2*sigma,(y_pred + 2*sigma)[::-1]]),
                alpha=.2, fc='orange', ec='None', label='95% Confidence Interval')

        ax[i][j].set_xlabel(r'$n1v$')
        ax[i][j].set_ylabel(r'IP Vibration Magnitude')
        ax[i][j].set_xlim(0.15, 1)
        ax[i][j].set_ylim(-0.8, 2)
        ax[i][j].scatter(abnormal['n1v_f'].values,abnormal['IP_Vib_Magnitude_A_f'].values,c = 'r',
                   label='Faulty',s=20,marker = 'x',linewidth = 0.8)
        ax[i][j].set_title(r"Lengthscale, $l$={0}   Smoothness, $\nu$ = {1}".format(l,nu))
        
#ax[i][j].legend(loc='lower left')
# Put a legend below current axis
ax[1][0].legend(loc='upper center', bbox_to_anchor=(1.12, -0.11),
          fancybox=True, ncol=5,fontsize=13)
