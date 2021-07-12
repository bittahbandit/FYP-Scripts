#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:38:19 2019


    - For each threshold, I perform cross validation
    
    - produces ROC curve from iterating through different thresholds

@author: thomasdrayton
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2
#%%

# Change to latex font ------------------------------------------------
plt.rcParams["font.family"] = "serif"


def check_data(x1,y1,clr1,size):
    '''
    Plot 2 datasets quickly
    '''
    fig,ax = plt.subplots()
    ax.scatter(x1,y1,c=clr1,s=size)
    #ax.scatter(x2,y2,c=clr2,s=size)



def cov_ellipse2(points, cov, nstd):
    """
    Source: https://stackoverflow.com/a/39749274/1391441
    """

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[::-1, 0]))

    # Confidence level
    q = 2 * norm.cdf(nstd) - 1                      
    r2 = chi2.ppf(q, 2)

    width, height = 2 * np.sqrt(vals * r2)

    return width, height, theta



def eigsorted(cov):
    '''
    Eigenvalues and eigenvectors of the covariance matrix.
    '''
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]



#%% Data Preparation: 
    
# features of interest
f1 = 'toilv_f'
f2 = 'IP_Vib_Magnitude_A_f'
# Remove unnecessary columns
keep = np.array([f1,f2,'Passed'])

dcol = np.setdiff1d(df_subset.columns.values,keep)
#dcol = ['n1v_f','tfuelv_f','tgtv_f','p30v_f']
df_subset.drop(dcol,axis=1,inplace=True)
#%%
# Separate data into normal and fauly
normal = df_subset.loc[df_subset['Passed'] == 1]
faulty = df_subset.loc[df_subset['Passed'] == 0]

# Faulty input and output data
X_f = faulty.loc[:,['toilv_f','IP_Vib_Magnitude_A_f']].values
y_f = faulty.loc[:,'Passed'].values

# Normal input and output data
X_n = normal.loc[:,['toilv_f','IP_Vib_Magnitude_A_f']].values
y_n = normal.loc[:,'Passed'].values

# Proportion of faulty data
fprop = len(faulty.index)/len(df_subset.index)

#% Synthetic Test Data 
# =============================================================================
# # Generate some data
# from sklearn.datasets.samples_generator import make_blobs
# X, y_true = make_blobs(n_samples=[34,1348], centers=[(2,3),(1.1,2)],
#                        cluster_std=0.40, random_state=0)
# 
# # Separate data into normal and fauly
# X_n = X[np.where(y_true == 1)]
# X_f = X[np.where(y_true == 0)]
# y_n = y_true[y_true == 1]
# y_f = y_true[y_true == 0]
# 
# # Faulty input and output data
# 
# fig,ax = plt.subplots()
# ax.scatter(X_n[:,0],X_n[:,1],s =10)
# ax.scatter(X_f[:,0],X_f[:,1],c ='r',s=5)
# 
# 
# #Proportion of faulty data
# fprop = len(X_f)/len(X)
# =============================================================================

#% Stratified kfold Cross Validation

#fig,ax = plt.subplots(1,2,squeeze=False,figsize=(14,5))

# Create data frame to store eval. metrics
cols = ['n_std','accuracy','FPR','TPR']
df_metrics = pd.DataFrame(columns=cols)

# Instantiate stratified kfold cross validation
k = 4
cv = StratifiedKFold(n_splits=k) 

count = 0
for n_std in np.linspace(0.04,3,297):

    # Classification accuracy scores, True Positive Rate, False Positive Rate
    accuracy = []
    tpsr = []
    fpsr = []
    
    
    idx = 0
    # Perform cross validation procedure
    for train, test in cv.split(X_n, y_n):
        
        # calculate proportion of faulty fp once + reshape to rows of 3 indices
        if(idx==0):
            fp = int(np.round((fprop*len(test))))
            f_smpl = np.reshape(np.array(range(k*fp)),(k,fp))             
            
            
            
        # Create current train and test data
        X_train = X_n[train, :]                                                 # from normal 
        X_test = np.concatenate( (X_f[f_smpl[idx], :] , X_n[test[fp+1:],:]) )   # from both classes
        y_train = y_n[train]
        y_test = np.concatenate( (y_f[f_smpl[idx]] , y_n[test[fp+1:]]) )        # from both classes
        
        
        # fit multivariate Gaussian distribution to obtain covariance & mean
        mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train, rowvar=0)
        
        # Find the dimensions of ellipse at different standard deviations
        width, height, theta = cov_ellipse2(X_train, cov, n_std)
        
        # Testing: find points in test set which lie within the ellipse
        cos_angle = np.cos(np.radians(180.-theta))
        sin_angle = np.sin(np.radians(180.-theta))
        
        xc = X_test[:,0] - mean[0]                                              # find data points position relative to ellipse center 
        yc = X_test[:,1] - mean[1]
        
        xct = xc * cos_angle - yc * sin_angle                                   # transform data points to align with major and minor axis
        yct = xc * sin_angle + yc * cos_angle 
        
        rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)               # equation of ellipse so all points are now transformed
        
        colours_array = []
        for r in rad_cc:
            if r <= 1.:
                # point in ellipse
                colours_array.append('darkblue')                                  
            else:
                # point not in ellipse
                colours_array.append('red') 
            
        
        if(count==100):
            f_rad_cc = rad_cc[0:np.sum(y_test==False)-1]
            n_rad_cc = rad_cc[np.sum(y_test==False)-1:]
            
            f_colours_array = []                                                      # separate points: in or out by colour
            n_colours_array = []
            f_markers = []
            n_markers = []
            for r in f_rad_cc:
                if r <= 1.:
                    # point in ellipse
                    f_colours_array.append('royalblue')                                  # darkblue point indicates predicted normal from model
                    f_markers.append('^')
                else:
                    # point not in ellipse
                    f_colours_array.append('orange')                                     # red point indicated predicted faulty from model
                    f_markers.append('^')
            for r in n_rad_cc:
                if r <= 1.:
                    # point in ellipse
                    n_colours_array.append('red')                                  # darkblue point indicates predicted normal from model
                    n_markers.append('o')
                else:
                    # point not in ellipse
                    n_colours_array.append('purple')
                    n_markers.append('o')
            
            
            # Plot data
            fig1,ax1 = plt.subplots()                                                 
            ax1.scatter(X_test[:,0][0:np.sum(y_test==False)-1],                      # plot faulty
                       X_test[:,1][0:np.sum(y_test==False)-1],
                       edgecolor=f_colours_array,
                       linewidth=0.7,
                       facecolors='none',
                       s=70,
                       marker='^')
            ax1.scatter(X_test[:,0][np.sum(y_test==False)-1:],                      # plot normal
                       X_test[:,1][np.sum(y_test==False)-1:],
                       color=n_colours_array,
                       linewidth=0.7,
                       s=55,
                       marker='1')
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                              edgecolor='k', fc='None', lw=0.9, zorder=4)
            ax1.add_patch(ellipse)
            ax1.set_ylabel('IP Vibration Magnitude',fontsize=12)
            ax1.set_xlabel('Oil Temperature',fontsize=12)
            
            # legend
            ax1.scatter([],[],label='FP'.format(f_colours_array.count('royalblue')),
                       edgecolor='royalblue',
                       linewidth=0.7,
                       facecolors='none',
                       s=70,
                       marker='^')
            
            ax1.scatter([],[],label='TN'.format(f_colours_array.count('orange')),
                       edgecolor='orange',
                       linewidth=0.7,
                       facecolors='none',
                       s=70,
                       marker='^')
            
            ax1.scatter([],[],label='TP'.format(n_colours_array.count('red')),
                       color='red',
                       linewidth=0.7,
                       s=55,
                       marker='1')
            ax1.scatter([],[],label='FN'.format(n_colours_array.count('purple')),
                       color='purple',
                       linewidth=0.7,
                       s=55,
                       marker='1')
            ax1.legend()
    #count+=1



        # Confusion matrix parameters
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for c,true in zip(colours_array,y_test):
            if((c=='red') and (true==0)):                                       # predicted faulty and is actually faulty
                TN+=1
            elif((c=='darkblue') and (true==1)):                                # predicted normal and is actually normal
                TP+=1
            elif((c=='red') and (true==1)):                                     # predicted faulty and is actually normal
                FN+=1
            else:                                                               # predicted normal and is actually faulty
                FP+=1
        
        # Store classification accuracy to test fold
        accuracy.append((TN+TP)/(TN+TP+FN+FP))
        
        # If ZeroDivisionError add NaN
        if(((FP==0) and (TN==0)) or ((TP==0) and (FP==0))):
            fpsr.append(np.nan)
            tpsr.append(np.nan)
            break
        else:                                                                   # store FPR and TPR scores from fold
            fpsr.append(FP/(FP+TN))
            tpsr.append(TP/(TP+FN))
        
        idx+=1
        
    
    # Mean scores 
    m_acc = np.array(accuracy).mean()
    m_fpsr = np.array(fpsr).mean()
    m_tpsr = np.array(tpsr).mean()
    
    # store means scores as row in metric dataframe
    df_metrics.loc[count] = [n_std,m_acc,m_fpsr,m_tpsr]
    
    #if(count==100):
     #   break
    
    count+=1
    
# Find AUC
    
auc = np.trapz(y=df_metrics.loc[:,'TPR'],x=df_metrics.loc[:,'FPR'],dx=0.01)
print(auc)
#%% Plot ROC curve:


ax[0][0].plot(np.linspace(0,1,100),np.linspace(0,1,100),linewidth=1.5,c='lightgrey')    # 0.5 reference line
ax[0][0].plot(df_metrics.loc[:,'FPR'],df_metrics.loc[:,'TPR'],linewidth = 1.5,
  label='k={0}, AUC={1:.2f}'.format(k,auc))         # 
ax[0][0].set_xlabel('FPR',fontsize = 12)
ax[0][0].set_ylabel('TPR',fontsize = 12)
ax[0][0].set_title('Fully-Labelled Performance Curve',fontsize=16)
ax[0][0].legend(fontsize=13,loc='lower right')

#%%


# features of interest
f1 = 'toilv_f'
f2 = 'IP_Vib_Magnitude_A_f'
# Remove unnecessary columns
keep = np.array([f1,f2,'Passed'])

dcol = np.setdiff1d(df_subset000.columns.values,keep)
#dcol = ['n1v_f','tfuelv_f','tgtv_f','p30v_f']
df_subset000.drop(dcol,axis=1,inplace=True)
#%%
# Separate data into normal and fauly
normal = df_subset000.loc[df_subset['Passed'] == 1]
faulty = df_subset000.loc[df_subset['Passed'] == 0]

# Faulty input and output data
X_f = faulty.loc[:,['toilv_f','IP_Vib_Magnitude_A_f']].values
y_f = faulty.loc[:,'Passed'].values

# Normal input and output data
X_n = normal.loc[:,['toilv_f','IP_Vib_Magnitude_A_f']].values
y_n = normal.loc[:,'Passed'].values

# Proportion of faulty data
fprop = len(faulty.index)/len(df_subset000.index)



# Create data frame to store eval. metrics
cols = ['n_std','accuracy','FPR','TPR']
df_metrics = pd.DataFrame(columns=cols)

# Instantiate stratified kfold cross validation
cv = StratifiedKFold(n_splits=k) 

count = 0
for n_std in np.linspace(0.04,3,297):

    # Classification accuracy scores, True Positive Rate, False Positive Rate
    accuracy = []
    tpsr = []
    fpsr = []
    
    
    idx = 0
    # Perform cross validation procedure
    for train, test in cv.split(X_n, y_n):
        
        # calculate proportion of faulty fp once + reshape to rows of 3 indices
        if(idx==0):
            fp = int(np.round((fprop*len(test))))
            f_smpl = np.reshape(np.array(range(k*fp)),(k,fp))             
            
            
            
        # Create current train and test data
        X_train = X_n[train, :]                                                 # from normal 
        X_test = np.concatenate( (X_f[f_smpl[idx], :] , X_n[test[fp+1:],:]) )   # from both classes
        y_train = y_n[train]
        y_test = np.concatenate( (y_f[f_smpl[idx]] , y_n[test[fp+1:]]) )        # from both classes
        
        
        # fit multivariate Gaussian distribution to obtain covariance & mean
        mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train, rowvar=0)
        
        # Find the dimensions of ellipse at different standard deviations
        width, height, theta = cov_ellipse2(X_train, cov, n_std)
        
        # Testing: find points in test set which lie within the ellipse
        cos_angle = np.cos(np.radians(180.-theta))
        sin_angle = np.sin(np.radians(180.-theta))
        
        xc = X_test[:,0] - mean[0]                                              # find data points position relative to ellipse center 
        yc = X_test[:,1] - mean[1]
        
        xct = xc * cos_angle - yc * sin_angle                                   # transform data points to align with major and minor axis
        yct = xc * sin_angle + yc * cos_angle 
        
        rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)               # equation of ellipse so all points are now transformed
        
        colours_array = []
        for r in rad_cc:
            if r <= 1.:
                # point in ellipse
                colours_array.append('darkblue')                                  
            else:
                # point not in ellipse
                colours_array.append('red') 

        # Confusion matrix parameters
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for c,true in zip(colours_array,y_test):
            if((c=='red') and (true==0)):                                       # predicted faulty and is actually faulty
                TN+=1
            elif((c=='darkblue') and (true==1)):                                # predicted normal and is actually normal
                TP+=1
            elif((c=='red') and (true==1)):                                     # predicted faulty and is actually normal
                FN+=1
            else:                                                               # predicted normal and is actually faulty
                FP+=1
        
        # Store classification accuracy to test fold
        accuracy.append((TN+TP)/(TN+TP+FN+FP))
        
        # If ZeroDivisionError add NaN
        if(((FP==0) and (TN==0)) or ((TP==0) and (FP==0))):
            fpsr.append(np.nan)
            tpsr.append(np.nan)
            break
        else:                                                                   # store FPR and TPR scores from fold
            fpsr.append(FP/(FP+TN))
            tpsr.append(TP/(TP+FN))
        
        idx+=1
        
    
    # Mean scores 
    m_acc = np.array(accuracy).mean()
    m_fpsr = np.array(fpsr).mean()
    m_tpsr = np.array(tpsr).mean()
    
    # store means scores as row in metric dataframe
    df_metrics.loc[count] = [n_std,m_acc,m_fpsr,m_tpsr]
    
    #if(count==100):
     #   break
    
    count+=1
    
# Find AUC
    
auc = np.trapz(y=df_metrics.loc[:,'TPR'],x=df_metrics.loc[:,'FPR'],dx=0.01)
print(auc)

ax[0][1].plot(np.linspace(0,1,100),np.linspace(0,1,100),linewidth=1.5,c='lightgrey')    # 0.5 reference line
ax[0][1].plot(df_metrics.loc[:,'FPR'],df_metrics.loc[:,'TPR'],linewidth = 1.5,
  label='k={0}, AUC={1:.2f}'.format(k,auc))         # 
ax[0][1].set_xlabel('FPR',fontsize = 12)
ax[0][1].set_ylabel('TPR',fontsize = 12)
ax[0][1].set_title('Hand-Labelled Performance Curve',fontsize=16)
ax[0][1].legend(fontsize=13,loc='lower right')