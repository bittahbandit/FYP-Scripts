#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:01:24 2019

    - Gaussian Process Regression [Modelling Normality]
    
    - Plotas figure with 5 subplots for the Preliminary analysis of 5 vars
    
    - Need to import IMPORT df_subset_p_curve_labelled BEFORE

@author: thomasdrayton
"""


import numpy as np
import matplotlib.pylab as plt
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


# Change to latex font ------------------------------------------------
plt.rcParams["font.family"] = "serif"


# normal and abnormal datasets
normal = df_subset.loc[df_subset['Passed'] == 1]
abnormal = df_subset.loc[df_subset['Passed'] == 0] 


gs = plt.GridSpec(3, 6,wspace=0.9)

fig = plt.figure(figsize=(10,21))
fig.subplots_adjust(left=0.09, bottom=0.1, right=0.99, top=0.89, wspace=0.3, hspace=0.3)

ax1 = fig.add_subplot(gs[0, 0:3]) #-----------------------------------------------


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
#% Convert to a numpy array
x = np.array(x)
y = np.array(y)
#% GP regression
kernel = ConstantKernel() + Matern(length_scale=1, nu=3/2) + WhiteKernel(noise_level=0.037)
X = x.reshape(-1, 1)
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)  # remove optimizer parameter for optimization
gp.fit(X, y)
print(gp.kernel_)
x_pred = np.linspace(0, 1,900).reshape(-1,1)
y_pred, sigma = gp.predict(x_pred, return_std=True)
ax1.scatter(x,y,label = "Normal",s=2)
ax1.plot(x_pred, y_pred, color='grey', label='Prediction')
ax1.fill(np.concatenate([x_pred, x_pred[::-1]]),
  np.concatenate([y_pred - 2*sigma,(y_pred + 2*sigma)[::-1]]),
  alpha=.2, fc='orange', ec='None', label='95% Confidence Interval')
ax1.set_xlabel(r'$n1v$')
ax1.set_ylabel(r'IP Vibration Magnitude')
ax1.set_xlim(0.15, 1)
ax1.set_ylim(-0.8, 1.18)
ax1.scatter(abnormal['n1v_f'].values,abnormal['IP_Vib_Magnitude_A_f'].values,c = 'r',
  label='Faulty',s=30,marker = 'x',linewidth = 0.8)
ax1.set_title(r"Lengthscale, $l$={0}   Smoothness, $\nu$ = {1}".format(0.102,3/2))


ax2 = fig.add_subplot(gs[0,3:]) #-----------------------------------------------

# 1D Array for x (n1v_f) and y (IP_vib) 
x = normal['n1v_f'].values
y = normal['tfuelv_f'].values
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
#% Convert to a numpy array
x = np.array(x)
y = np.array(y)
#% GP regression
kernel = ConstantKernel() + Matern(length_scale=1, nu=3/2) + WhiteKernel(noise_level=0.037)
X = x.reshape(-1, 1)
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)  # remove optimizer parameter for optimization
gp.fit(X, y)
print(gp.kernel_)
x_pred = np.linspace(0, 1,900).reshape(-1,1)
y_pred, sigma = gp.predict(x_pred, return_std=True)
ax2.scatter(x,y,label = "Normal",s=2)
ax2.plot(x_pred, y_pred, color='grey', label='Prediction')
ax2.fill(np.concatenate([x_pred, x_pred[::-1]]),
  np.concatenate([y_pred - 2*sigma,(y_pred + 2*sigma)[::-1]]),
  alpha=.2, fc='orange', ec='None', label='95% Confidence Interval')
ax2.set_xlabel(r'$n1v$')
ax2.set_ylabel(r'Fuel Temperature')
ax2.set_xlim(0.16, 1.01)
ax2.set_ylim(0.32, 1.15)
ax2.scatter(abnormal['n1v_f'].values,abnormal['tfuelv_f'].values,c = 'r',
  label='Faulty',s=30,marker = 'x',linewidth = 0.8)
ax2.set_title(r"Lengthscale, $l$={0}   Smoothness, $\nu$ = {1}".format(0.109,3/2))


ax3 = fig.add_subplot(gs[1,0:3]) #-----------------------------------------------


# 1D Array for x (n1v_f) and y (IP_vib) 
x = normal['n1v_f'].values
y = normal['toilv_f'].values
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
#% Convert to a numpy array
x = np.array(x)
y = np.array(y)
#% GP regression
kernel = ConstantKernel() + Matern(length_scale=1, nu=3/2) + WhiteKernel(noise_level=0.037)
X = x.reshape(-1, 1)
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)  # remove optimizer parameter for optimization
gp.fit(X, y)
print(gp.kernel_)
x_pred = np.linspace(0, 1,900).reshape(-1,1)
y_pred, sigma = gp.predict(x_pred, return_std=True)
ax3.scatter(x,y,label = "Normal",s=2)
ax3.plot(x_pred, y_pred, color='grey', label='Prediction')
ax3.fill(np.concatenate([x_pred, x_pred[::-1]]),
  np.concatenate([y_pred - 2*sigma,(y_pred + 2*sigma)[::-1]]),
  alpha=.2, fc='orange', ec='None', label='95% Confidence Interval')
ax3.set_xlabel(r'$n1v$')
ax3.set_ylabel(r'Oil Temperature')
ax3.set_xlim(0.15, 1.01)
ax3.set_ylim(0.3, 1.01)
ax3.scatter(abnormal['n1v_f'].values,abnormal['toilv_f'].values,c = 'r',
  label='Faulty',s=30,marker = 'x',linewidth = 0.8)
ax3.set_title(r"Lengthscale, $l$={0}   Smoothness, $\nu$ = {1}".format(0.156,3/2))


ax4 = fig.add_subplot(gs[1,3:])#-----------------------------------------------


# 1D Array for x (n1v_f) and y (IP_vib) 
x = normal['n1v_f'].values
y = normal['tgtv_f'].values
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
#% Convert to a numpy array
x = np.array(x)
y = np.array(y)
#% GP regression
kernel = ConstantKernel() + Matern(length_scale=1, nu=3/2) + WhiteKernel(noise_level=0.037)
X = x.reshape(-1, 1)
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)  # remove optimizer parameter for optimization
gp.fit(X, y)
print(gp.kernel_)
x_pred = np.linspace(0, 1,900).reshape(-1,1)
y_pred, sigma = gp.predict(x_pred, return_std=True)
ax4.scatter(x,y,label = "Normal",s=2)
ax4.plot(x_pred, y_pred, color='grey', label='Prediction')
ax4.fill(np.concatenate([x_pred, x_pred[::-1]]),
  np.concatenate([y_pred - 2*sigma,(y_pred + 2*sigma)[::-1]]),
  alpha=.2, fc='orange', ec='None', label='95% Confidence Interval')
ax4.set_xlabel(r'$n1v$')
ax4.set_ylabel(r'Turbine Gas Temperature')
ax4.set_xlim(0.1, 1.01)
ax4.set_ylim(0.1, 1.01)
ax4.scatter(abnormal['n1v_f'].values,abnormal['tgtv_f'].values,c = 'r',
  label='Faulty',s=30,marker = 'x',linewidth = 0.8)
ax4.set_title(r"Lengthscale, $l$={0}   Smoothness, $\nu$ = {1}".format(0.0931,3/2))


ax5 = fig.add_subplot(gs[2,0:3])#-----------------------------------------------

# 1D Array for x (n1v_f) and y (IP_vib) 
x = normal['n1v_f'].values
y = normal['p30v_f'].values
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
#% Convert to a numpy array
x = np.array(x)
y = np.array(y)
#% GP regression
kernel = ConstantKernel() + Matern(length_scale=1, nu=3/2) + WhiteKernel(noise_level=0.037)
X = x.reshape(-1, 1)
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)  # remove optimizer parameter for optimization
gp.fit(X, y)
print(gp.kernel_)
x_pred = np.linspace(0, 1,900).reshape(-1,1)
y_pred, sigma = gp.predict(x_pred, return_std=True)
ax5.scatter(x,y,label = "Normal",s=2)
ax5.plot(x_pred, y_pred, color='grey', label='Prediction')
ax5.fill(np.concatenate([x_pred, x_pred[::-1]]),
  np.concatenate([y_pred - 2*sigma,(y_pred + 2*sigma)[::-1]]),
  alpha=.2, fc='orange', ec='None', label='95% Confidence Interval')
ax5.set_xlabel(r'$n1v$')
ax5.set_ylabel(r'Compressor Pressure')
ax5.set_xlim(0.157, 1.01)
ax5.set_ylim(-0.05, 1.01)
ax5.scatter(abnormal['n1v_f'].values,abnormal['p30v_f'].values,c = 'r',
  label='Faulty',s=30,marker = 'x',linewidth = 0.8)
ax5.set_title(r"Lengthscale, $l$={0}   Smoothness, $\nu$ = {1}".format(0.141,3/2))

fig.text(0.52, 0.94, 'Discordancy Test for Variables in Subset', ha='center',fontsize=15)

ax5.legend(loc='upper center', bbox_to_anchor=(1.62, 0.7),fancybox=True,fontsize=12)



fig.savefig('/Users/thomasdrayton/Desktop/discordancy_test_fl.png', format='png', dpi=280)


#%%
#fig.subplots_adjust(left=0.09, bottom=0.1, right=0.99, top=0.89, wspace=0.3, hspace=0.3)
ax5.legend(loc='upper center', bbox_to_anchor=(1.62, 0.7),fancybox=True,fontsize=12)

fig.show()
