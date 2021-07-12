#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:46:08 2018

    - Plots n1v vs. IP_vib 

    - IMPORT df_subset_p_curve_labelled BEFORE
    
    - Run section by section  &  read section before you run it.

@author: thomasdrayton
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


#%% Remove all unneeded columns
df_subset.drop(labels =['tfuelv_f','toilv_f','tgtv_f','p30v_f'],axis = 1, inplace = True)

#%% Creating list where each element contains a dataframe where all the indices are from a sinlge window

windows_df_list = []
for w in range(6):
    windows_df_list.append(df_subset.loc(axis=0)[:,w])  # - indexes all the engine tests, of each window
#%% 
# normal and abnormal datasets
normal = df_subset.loc[df_subset['Passed'] == 1]
abnormal = df_subset.loc[df_subset['Passed'] == 0] 

#%% Plotting results all n1v IP_vib points  -  color coded by step


fig,ax = plt.subplots()

idx = 0
colours = ['k','r','royalblue','darksalmon','mediumseagreen','orange']
for i,colour in zip(windows_df_list,colours):
    ax.scatter(i.loc[:,'n1v_f'],i.loc[:,'IP_Vib_Magnitude_A_f'],c = colour,s = 12)

#%% Plotting individual steps with annotation 

step = 3
    
fig,ax = plt.subplots(1,1,figsize =(16,16))
ax.scatter(windows_df_list[step].loc[:,'n1v_f'],windows_df_list[step].loc[:,'IP_Vib_Magnitude_A_f'],c = 'k',s = 2)
ax.scatter(abnormal['n1v_f'].values,abnormal['IP_Vib_Magnitude_A_f'].values,c = 'r',
           label='Abnormal',s=30,marker = 'x',linewidth = 0.5)

    
for row in windows_df_list[step].iterrows():
     ax.annotate(row[0],(row[1][0], row[1][1]),fontsize = 5)

#%%
IP_vib_n1v_step3_mixture = windows_df_list[3][(windows_df_list[3].n1v_f>=0.804) & (windows_df_list[3].n1v_f<=0.815) & (windows_df_list[3].IP_Vib_Magnitude_A_f>=0.945) & (windows_df_list[3].IP_Vib_Magnitude_A_f<=0.99)]

#%% 1D Array for x (n1v_f) and y (IP_vib) 
x = normal['n1v_f'].values
y = normal['IP_Vib_Magnitude_A_f'].values

#%% Reorganise x and y values for normal data
# Create list of tuples of xy
xy = []
for i in range(len(x)):
    xy.append((x[i],y[i]))
    
# Sort list so that ascends by first element in tuple
xy = sorted(xy,key=lambda x : x[0])

# Put them back into individual arrays
x= []
y = []
for i in xy:
    x.append(i[0])
    y.append(i[1])


#%%

x = np.array(x)
# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
#plt.plot(x, y, color='cornflowerblue', linewidth=lw,
#         label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([7, 8, 9]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)
    print(degree,mean_squared_error(y,y_plot))


plt.legend(loc='lower left')

plt.show()
#%%  Fitting curve to data - 3x3 subplots of differen polynomial orders

# set starting order and number of nxn plots 
order = 1

num_nn_plts = 3

xp = np.linspace(0.16, 1,100)

# subplots

fig,ax = plt.subplots(num_nn_plts,num_nn_plts,figsize = (16,16))

for n in range(num_nn_plts):
    for k in range(num_nn_plts):
        
        z = np.polyfit(x,y,order)
        p = np.poly1d(z)

        ax[n,k].scatter(x,y,label = "Normal",s=1)
        ax[n,k].plot(xp,p(xp),label = "Polynomial with order={}".format(order),
          color='C1')
        ax[n,k].scatter(abnormal['n1v_f'].values,abnormal['IP_Vib_Magnitude_A_f'].values,c = 'r',
           label='Abnormal',s=15,marker = 'x',linewidth = 0.5)
        
        # Plot settings
        ax[n,k].legend(fontsize = 5,loc='upper left',
          fancybox=True)
        ax[n,k].set_xlim([0.16,1.02])
        ax[n,k].set_ylim([0,1.05])
        ax[n,k].tick_params(axis = 'both',labelsize = 8)
        print(order,mean_squared_error(y,p(x)))
        order+=1
        


fig.text(0.5, 0.04, 'n1v', ha='center')
fig.text(0.04, 0.5, 'IP vibration', va='center', rotation='vertical')


# =============================================================================
# fig.savefig("/Users/thomasdrayton/Desktop/Choosing_polynomial_order _for_IPvib_n1v_curve_fitting_v3.png",bbox_inches='tight',dpi = 600) 
# =============================================================================
#%% 1 Plot
order = 12
xp = np.linspace(0.16, 1,100)


fig,ax = plt.subplots(1,1,figsize = (16,16))

z = np.polyfit(x,y,order)
p = np.poly1d(z)
ax.scatter(x,y,label = "Normal",s=1)
ax.plot(xp,p(xp),label = "Polynomial fitted to normal data".format(order),color='C1')
ax.scatter(abnormal['n1v_f'].values,abnormal['IP_Vib_Magnitude_A_f'].values,c = 'r',
  label='Abnormal',s=15,marker = 'x',linewidth = 0.8)
        
# Plot settings
ax.legend(fontsize = 10,loc='upper left',fancybox=True)
ax.set_xlim([0.16,1.02])
ax.set_ylim([0,1.05])
ax.tick_params(axis = 'both',labelsize = 8)
#%%
print(mean_squared_error(y,p(x)))

#%% Plotting ± rolling std deviation 

# Convert y array back to series
y = pd.Series(y)

# intialise forst window size
w  = 10

# number of nxn plots 
num_nn_plts = 2

# x values to supply to polynomial for curve fitting
xp = np.linspace(0.16, 1,len(x))

# PLot 3x3
fig,ax = plt.subplots(num_nn_plts,num_nn_plts,figsize = (16,16))

for n in range(num_nn_plts):
    for k in range(num_nn_plts):
        
        # Fitting curve as 8th order polynomial 
        z = np.polyfit(x,y.values,8)
        p = np.poly1d(z)
        
        
        # Calculate standard devation
        std_dev = y.rolling(w,center= True).std() 
        
        ax[n,k].scatter(x,y.values,label = "Normal",s=1)  # plot data points
        ax[n,k].plot(xp,p(xp),label = "8th order polynomial",
          color='C1') # plotting curve
        ax[n,k].plot(xp,p(xp)+std_dev,c='y',label=" ±1 standard deviation w/ "+ 
          "Rolling window size={}".format(w))
        ax[n,k].plot(xp,p(xp)-std_dev,c='y')
        #plotting abnormal
        ax[n,k].scatter(abnormal['n1v_f'].values,abnormal['IP_Vib_Magnitude_A_f'].values,c = 'r',
           label='Abnormal',s=15,marker = 'x',linewidth = 0.5)
        ax[n,k].legend(fontsize = 5,loc='upper left',
          fancybox=True)
        ax[n,k].tick_params(axis = 'both',labelsize = 7)

        
        w+=20  # increment window size



fig.text(0.5, 0.04, 'n1v', ha='center')
fig.text(0.04, 0.5, 'IP vibration', va='center', rotation='vertical')


#fig.savefig("/Users/thomasdrayton/Desktop/Different Rolling windows standard deviation.png",bbox_inches='tight',dpi = 600) 


#%% Separating data points into windows

# Window boundaries
bndrs = [0,.25,.32,.45,.55,.7,.78,.83,.9,.96,1]

# Separating data points into windows

windows = []  # list of arrays containing separated data points
windows_n1v = []
windows_IP = []

for i in range(len(bndrs)):
    w = []
    windows.append(w)
    n = []
    windows_n1v.append(n)
    IP = []
    windows_IP.append(IP)
    for ii in xy:
        if((ii[0]>bndrs[i]) and ii[0]<=bndrs[i+1]):
            w.append(ii)
            n.append(ii[0])
            IP.append(ii[1])

del windows[-1]
del windows_n1v[-1]
del windows_IP[-1]

#%% standard deviation of each window

std_dev = []
for i in windows_IP:
    std_dev.append(np.std(i))
    
#%% Average value of n1v in each window

n1v_avg = []
for i in windows_n1v:
    n1v_avg.append(round(np.average(i),2))

#%% Plot standard deviation points above line


xp = np.linspace(0.16, 1,100)

# Fitting curve to data
z = np.polyfit(x,y,12) 
p = np.poly1d(z)


plus = []
minus = []
for i in enumerate(n1v_avg):
    plus.append(p(i[1])+std_dev[i[0]])
    minus.append(p(i[1])-std_dev[i[0]]) 
#%%
# Fitting curve to ± standard devations 
z = np.polyfit(n1v_avg,plus,8)
p_up = np.poly1d(z)

z = np.polyfit(n1v_avg,minus,8)
p_down = np.poly1d(z)


#%% Plotting ± 1 standard devations by creating windows 
fig,ax = plt.subplots(1,1,figsize = (16,16))

ax.scatter(x,y,label = "Normal",s=1)  # plot data points
ax.plot(xp,p(xp),label = "Polynomial fitted to normal data",color='C1') # plotting curve
ax.scatter(n1v_avg,plus,marker = '_',s = 200,c='k',label='± 1 standard deviation',linewidth = 0.7)
ax.scatter(n1v_avg,minus,marker = '_',s = 200,c='k',linewidth = 0.7)
#ax.plot(xp,p_up(xp),label=" ±1 standard deviation",color='k') # plotting upper curve
#ax.plot(xp,p_down(xp),color='k') # plotting lpwer curve
ax.scatter(abnormal['n1v_f'].values,abnormal['IP_Vib_Magnitude_A_f'].values,c = 'r',
           label='Abnormal',s=15,marker = 'x',linewidth = 0.5)
ax.set_xlim([0.16,1.02])
ax.set_ylim([0,1.05])
ax.legend()

fig.text(0.5, 0.04, 'n1v', ha='center')
fig.text(0.04, 0.5, 'IP vibration', va='center', rotation='vertical')

