#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:04:44 2019

    - gmm_CI_untidy
    
    - fits bivariate distribution to data labelled as normal then tests
      for both normal and faulty classes using stratified cross validation
      and produces matrics for accuracy, FPR and TPR for ROC curve
    
    - made for fitting gaussinan to toilv - IP data
    
    - import df_subset for hand labelled data set
    - import df_subset_all_f_win for labelling of every step in performance curve 


@author: thomasdrayton
"""




import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2
from random import randint
#%%


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

print(2 * norm.cdf(1.2) - 1)

#%% Data Preparation: 

# Remove unnecessary columns
dcol = ['n1v_f','tfuelv_f','tgtv_f','p30v_f']
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

#%% Test Data 
# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=[34,1348], centers=[(2,3),(0.3,2)],
                       cluster_std=0.40, random_state=0)

# Separate data into normal and fauly
X_n = X[np.where(y_true == 1)]
X_f = X[np.where(y_true == 0)]
y_n = y_true[y_true == 1]
y_f = y_true[y_true == 0]

# Faulty input and output data

fig,ax = plt.subplots()
ax.scatter(X_n[:,0],X_n[:,1],s =10)
ax.scatter(X_f[:,0],X_f[:,1],c ='r',s=5)


#Proportion of faulty data
fprop = len(X_f)/len(X)

#%% Stratified kfold Cross Validation

# Create data frame to store eval. metrics
cols = ['n_std','accuracy','FPR','TPR']
df_metrics = pd.DataFrame(columns=cols)

# Instantiate stratified kfold cross validation
cv = StratifiedKFold(n_splits=10) 


# CHOOSE
n_std= 1.2

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
        f_smpl = np.reshape(np.array(range(len(test))),(-1,fp))             # am I shuffling here??



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
    
    rad_cc = (xct**2/(width/2.)**2) + (yct**2/(height/2.)**2)               # equation of ellipse
    
    colours_array = []                                                      # separate points: in or out by colour
    for r in rad_cc:
        if r <= 1.:
            # point in ellipse
            colours_array.append('darkblue')                                # darkblue point indicates predicted normal from model
        else:
            # point not in ellipse
            colours_array.append('red')                                     # red point indicated predicted faulty from model
    
    # Plot data
    fig,ax = plt.subplots()                                                 
    ax.scatter(X_test[:,0],X_test[:,1],c=colours_array,linewidth=0.3,s=5)   # plots test set points that lie in ellispe red
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                       edgecolor='r', fc='None', lw=.8, zorder=4)
    ax.add_patch(ellipse)
        
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
        elif((c=='darkblue') and (true==0)):                                # predicted normal and is actually faulty
            FP+=1
        else:                                                               # predicted faulty and is actually normal
            FN+=1
    
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
    
    if(idx==0):
        break
    
    idx+=1

#%% Testing confidence ellipse code

def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation

# Generate some data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=[600], centers=[(3,3)],
                       cluster_std=0.50, random_state=0)

# plot raw data
fig,ax = plt.subplots()
ax.scatter(X[:,0],X[:,1],s =10)

mean = np.mean(X, axis=0)
cov = np.cov(X, rowvar=0)


# plotting 
for i in np.linspace(0.1,1,10):
    
    w,h,r= cov_ellipse(cov=cov,q=i)
    
    ellipse = Ellipse(xy=mean, width=w, height=h, angle=r,
                            edgecolor='r', fc='None', lw=.8, zorder=4,label=i)
    ax.add_patch(ellipse)

ax.legend()
