#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:46:08 2018

    - import df_subset.spydata

    - Fisher's Linear Discriminant on cols
    
    
@author: thomasdrayton
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import math


# Change to latex font ------------------------------------------------
plt.rcParams["font.family"] = "serif"


df = df_subset                                                                      # CHANGE 
#%% save labelled data and drop Passed column in preparation for dim. reduction
passed = df.Passed


#%% Select ALL variables for dimensionality reduction

cols = ['n1v_f','tfuelv_f','toilv_f','LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
        'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
        'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
        'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

cols = ['tfuelv_f','toilv_f','LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
        'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
        'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
        'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

#%% Select a subset of variables

# original subset
#cols = ['n1v_f','tfuelv_f','toilv_f','IP_Vib_Magnitude_A_f','tgtv_f','p30v_f']




fig2, ax2 = plt.subplots(2,figsize=(6.8,7))

fig2.text(0.52, 0.96, 'Optimal EFS Subset: Fully-Labelled Performance Curve', ha='center',fontsize=10)               # CHANGE 0.3  0.73


sp = 0
#SFS fully labelled subsets
#cols = ['tcarv_S_f','toilv_f','P44_f','tsasv_S_f','tprtrimmed_f'] #kappa k1
#cols = ['tcarv_S_f','t25v_f','t20v_f','trav_f','tsasv_S_f','p30v_f','t30v_f','tprtrimmed_f'] #kappa k3
#cols = ['t20v_f','P44_f','p30v_f','IP_Vib_Magnitude_A_f','tsasv_S_f','trav_f','ps26v_f','tgtv_f'] #acc k1
#cols = ['tcarv_S_f','t25v_f','t20v_f','trav_f','t30v_f','p30v_f','tcafv_S_f','P44_f','P42_f'] #acc k3

# SFS hand labelled subsets
#cols = ['IP_Vib_Magnitude_A_f','t20v_f','tsasv_S_f','tprtrimmed_f','t25v_f','P42_f','ps26v_f']#kappa k1
#cols = ['P44_f','t25v_f','IP_Vib_Magnitude_A_f','t20v_f','tsasv_S_f','trav_f','P42_f','p30v_f','tprtrimmed_f']#kappa k3
#cols = ['IP_Vib_Magnitude_A_f','P44_f','t25v_f','ps26v_f','trav_f','tprtrimmed_f','t20v_f','P42_f']#acc k1
#cols = ['P44_f','t25v_f','IP_Vib_Magnitude_A_f','t20v_f','tprtrimmed_f','trav_f','tsasv_S_f']#acc k3

#EFS - fully labelled
#cols = ['t20v_f','t30v_f','p20v_f','ps26v_f','tgtv_f','poilv_S_f']

#EFS hand labelled
#cols = ['IP_Vib_Magnitude_A_f', 't20v_f', 't25v_f', 't30v_f', 'p30v_f', 'P44_f', 'trav_f', 'tsasv_S_f']

# create data that PCA will be performed on 
X = df.loc[:,cols].values
y = df.loc[:,'Passed'].values


#%%
# =============================================================================
# # instantiante PCA object - reduction to 2 principle components
# lda = LinearDiscriminantAnalysis(n_components=2)
# lda_h = lda.fit_transform(X,y)
# 
# # Create histogram
# 
# min_b = math.floor(np.min(lda_h))
# max_b = math.ceil(np.max(lda_h))
# bins = np.linspace(min_b, max_b, 200) 
# 
# df_lda_hist = pd.DataFrame()
# 
# df_lda_hist["PC1"] = lda_h[:,0]
# df_lda_hist["y"] = y
# 
# normal_h  = df_lda_hist.loc[df_lda_hist.y == 1]
# faulty_h = df_lda_hist.loc[df_lda_hist.y == 0]
# 
# fig, ax = plt.subplots(1,1,figsize=(16,10))
# ax.hist(normal_h.PC1,label='Formal',bins=bins)
# ax.hist(faulty_h.PC1,label='Faulty',bins=bins,color='r')
# ax.set_title('FDA histogram ')
# =============================================================================


#%% Label data

def comp_mean_vectors(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y==cl], axis=0))
    return mean_vectors

def scatter_within(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_W = np.zeros((n_features, n_features))
    for cl, mv in zip(class_labels, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))                 
        for row in X[y == cl]:
            row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat                           
    return S_W

def scatter_between(X, y):
    overall_mean = np.mean(X, axis=0)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)    
    S_B = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):  
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_B


S_W, S_B = scatter_within(X, y), scatter_between(X, y)

#%%
e_vals, e_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

#print('Eigenvectors \n%s' %e_vecs)
#print('\nEigenvalues \n%s' %e_vals)

# Make a list of (eigenvalue, eigenvector) tuples
e_pairs = [(np.abs(e_vals[i]), e_vecs[:,i]) for i in range(len(e_vals))]


# Sort the (eigenvalue, eigenvector) tuples from high to low
e_pairs.sort(key=lambda x: x[0])
e_pairs.reverse()

#print('Eigenvalues in descending order:')
#for i in e_pairs:
#    print(i[0]) #print eigen values

total = sum(e_vals)
var_exp = [(i / total)*100 for i in sorted(e_vals, reverse=True)]
#print(var_exp)

W = np.hstack((e_pairs[0][1].reshape(X.shape[1],1), e_pairs[1][1].reshape(X.shape[1],1)))
#print('Matrix W:\n', W.real)

#%%

# Transforming the samples onto the new subspace
X_lda = X.dot(W)


df_lda = pd.DataFrame()

df_lda["PC1"] = X_lda[:,0]
df_lda["PC2"] = X_lda[:,1]
df_lda["y"] = y

normal  = df_lda.loc[df_lda.y == 1]
faulty = df_lda.loc[df_lda.y == 0]
#%%
# Histogram for PC1
min_b = np.min(df_lda.PC1.values)
max_b = np.max(df_lda.PC1.values)
bins = np.linspace(min_b, max_b, 200) 

fig, ax = plt.subplots(1,1)
ax.hist(normal.PC1,label='Formal',bins=bins)
ax.hist(faulty.PC1,label='Faulty',bins=bins,color='r')
ax.set_title('FLD x-axis histogram density estimation')
ax.legend()

# Histogram PC2
min_b = np.min(df_lda.PC2.values)
max_b = np.max(df_lda.PC2.values)
bins = np.linspace(min_b, max_b, 200) 

fig1, ax1 = plt.subplots(1,1)
ax1.hist(normal.PC2,label='Formal',bins=bins)
ax1.hist(faulty.PC2,label='Faulty',bins=bins,color='r')
ax1.set_title('FLD y-axis histogram density estimation ')
ax1.legend()


# Fisher's Linear Discriminant 
fig2.subplots_adjust(top=0.95, wspace=0.3, hspace=0.3,bottom=0.09)
ax2[sp].scatter(normal.PC1,normal.PC2,s = 2,label='Normal')
ax2[sp].scatter(faulty.PC1,faulty.PC2,c = 'r',
           label='Faulty',s=35,marker = 'x',linewidth = 0.7)
#ax2[sp].set_title("")


ax2[sp].legend()

#%%
#fig2.savefig('/Users/thomasdrayton/Desktop/FLD_fl_hl_efs.png',dpi=400,format='png')
#%%
# =============================================================================
# X = np.array([(1,2),(2,3),(3,3),(4,5),(5,5),(1,0),(2,1),(3,1),(3,2),(5,3),(6,5)])
# y = np.array([0,0,0,0,0,1,1,1,1,1,1])
# 
# plt.scatter(X[0:5,0],X[0:5,1])
# plt.scatter(X[5:,0],X[5:,1],marker='x')
# 
# 
# # separate into classes
# X1 = X[np.where(y == 0)]
# X2 = X[np.where(y == 1)]
# 
# # means
# m1_x = np.mean(X1[:,0])
# m1_y = np.mean(X1[:,1])
# m1 = np.vstack((m1_x,m1_y))
# 
# m2_x = np.mean(X2[:,0])
# m2_y = np.mean(X2[:,1])
# m2 = np.vstack((m2_x,m2_y))
# 
# 
# # covariances
# # cov1
# cov1 = []
# for i in X1:
#     cov1.append((np.dot(i.reshape(2,1) - m1,(i.reshape(2,1) - m1).transpose())))
#     
# cov1 = (sum(cov1)/(len(cov1)-1))*(len(cov1)-1)
#     
# # cov1
# cov2 = []
# for i in X2:
#     cov2.append((np.dot(i.reshape(2,1) - m2,(i.reshape(2,1) - m2).transpose())))
#     
# cov2 = (sum(cov2)/(len(cov2)-1))*(len(cov2)-1)
#     
# #%
# # scatter within
# SW = cov1 + cov2
# 
# # scatter between
# SB = (m1 - m2)*(m1 - m2).transpose()
# 
# invSW = np.linalg.inv(SW)
# 
# invSW_by_SB = invSW*SB
# 
# # get projection vector
# #D, V = np.linalg.eig(invSW_by_SB)
# W = np.dot(invSW,(m1 - m2))
# 
# #projection vector
# #W = V[:,0]
# 
# #%%
# 
# X_fld = np.dot(W.transpose(),np.transpose(X1))
# 
# #plt.scatter(X_fld[0:5,0],X_fld[0:5,1])
# #plt.scatter(X_fld[5:,0],X_fld[5:,1],marker='x')
# print(X_fld)
# =============================================================================
