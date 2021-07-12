#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:43:17 2019

    - Individual variable analysis that forms a 3x1 plot 

    - Actual Sequential Forward Feature Selection

@author: thomasdrayton
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import balanced_accuracy_score
from difflib import SequenceMatcher

# ignore divide by zero error
np.seterr(divide='ignore')

# Change to latex font ------------------------------------------------
plt.rcParams["font.family"] = "serif"

variables = ['tfuelv_f','toilv_f','LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
        'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
        'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
        'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

# Dataframe to store metric
df_metrics = pd.DataFrame(columns = ['Variable Set','Accuracy','Kappa','MCC','TNR','TPR','Miss_Rate','FPR','k_neighbours','Variable'])

i_ft = 0
# iterating through different k values in knn
for nn in [1,3,5,7]:
    # Instantiate kNN classifier
    knn = KNeighborsClassifier(n_neighbors = nn,metric='cosine')
    
    # Instantiate stratified kfold cross validation
    k = 10
    cv = StratifiedKFold(n_splits=k) 
    
    # output labels - same for each feature value
    y = df_subset.loc[:,'Passed'].values
    
    #i_ft = 0
    for ft in variables:
        
        # Prepare data X,y
        X = df_subset.loc[:,ft].values
        
        # lists to store metrics for each fold to eventually average
        acc = []
        kappa = []
        tnr = []
        tpr = []
        mcc = []
        miss_rate = []
        fpr = []
    
        idx = 0
        for train, test in cv.split(X, y):
            #print(train,test)
                              
            # Create current train and test data
            X_train = X[train].reshape(-1,1)
            y_train = y[train]
            X_test = X[test].reshape(-1,1)
            y_test = y[test]
            
            # Train kNN / store data point values and corresponding class from training set
            knn.fit(X_train,y_train)
            
            # Test
            y_pred = knn.predict(X_test)
            
            
            # --------------------------------------------------------------------
            # Plot the decision boundary
            #plot_predict_proba()
            
            # Confusion matrix
            TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()
            total = TN+FP+FN+TP 
        
            # Cohen's Kappa Coefficient 
            exp = ((TN+FP)*(TN+FN) + (FN+TP)*(FP+TP))/(total*total)
            obs = (TP + TN) / (TP + FN + FP + TN)
            cohen_kappa = (obs - exp) / (1 - exp)
            kappa.append(cohen_kappa)
            
            
            # Harmonic accuracy 
            acc.append(2/(((TP+FN)/TP)+((TN+FP)/TN)))
            
            # MCC
            mcc.append((TP*TN - FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))))
            
            # True negative rate / specificity & sensitivity
            tnr.append(TN/(TN+FP))      # assess faulty data
            tpr.append(TP/(TP+FN))   # using sensitivity because it's there to assess normal data
            miss_rate.append(FN/(FN+TP))
            fpr.append(FP/(FP+TN))
        
    
        df_metrics.loc[i_ft,"k_neighbours"] = nn
        df_metrics.loc[i_ft,"Variable"] = ft
        df_metrics.loc[i_ft,"Accuracy"] = np.mean(acc)
        df_metrics.loc[i_ft,'TNR'] = np.mean(tnr)
        df_metrics.loc[i_ft,'TPR'] = np.mean(tpr)
        df_metrics.loc[i_ft,'Miss_Rate'] = np.mean(miss_rate)
        df_metrics.loc[i_ft,'FPR'] = np.mean(fpr)
        df_metrics.loc[i_ft,"Kappa"] = np.mean(kappa)
        df_metrics.loc[i_ft,"MCC"] = np.mean(mcc)
        df_metrics.loc[i_ft,"Variable Set"] = ft



        
    
    
        i_ft+=1
    

# sort variable
sorting_v = 'Accuracy'                                                             # CHANGE


# Sorting for 1 and 3
df_metrics_k1 = df_metrics[df_metrics.k_neighbours==1]                          
df_metrics_k1.sort_values(by=sorting_v,inplace = True,ascending=False)
df_metrics_k1 = df_metrics_k1.reset_index(drop=True)
#df_metrics_k1.to_excel('/Users/thomasdrayton/Desktop/FYP/Final Report/Individual Variable Analysis/HL/univarite_rank_k1_hl.xlsx')

df_metrics_k3 = df_metrics[df_metrics.k_neighbours==3]                          
df_metrics_k3.sort_values(by=sorting_v,inplace = True,ascending=False)
df_metrics_k3 = df_metrics_k3.reset_index(drop=True)
#df_metrics_k3.to_excel('/Users/thomasdrayton/Desktop/FYP/Final Report/Individual Variable Analysis/HL/univarite_rank_k3_hl.xlsx')


# =============================================================================
# #%% Plot all bar chart of Harmonic mean accuracy for all variables k_neighbours 1,3,5,7
# # Plot comparison of k for ranking
# fig,ax = plt.subplots(3,1,figsize = (6.2,30))
# 
# # New df to calculate the mean Kappa statistic for Youden's J statistic
# df_kappa = pd.DataFrame(index=[7,5,3,1],columns = ['k-Neighbours','Mean Accuracy','Max','Min']) 
# 
# for i in [7,5,3,1]:
#     ax[1].bar(x=df_metrics[df_metrics.k_neighbours==i].loc[:,'Variable'].values,
#                         height=df_metrics[df_metrics.k_neighbours==i].loc[:,'Accuracy'],
#                         align='center',
#                         width=0.5,
#                         color='darkblue',
#                         label=r'$k$-neighbours:{0}'.format(i),
#                         alpha=i/10)                        #
#     
#     # rotate tick labels               
#     #for tick in ax.get_xticklabels():
#      #   tick.set_rotation(90)
#       #  tick.set_fontsize(18)
#     
#     # find total total Kappa Statisitc for each k_neighbour
#     df_kappa.loc[i,'k-Neighbours'] = i
#     df_kappa.loc[i,'Mean Accuracy'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'Accuracy'].values.mean()
#     df_kappa.loc[i,'Max'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'Accuracy'].values.max()
#     df_kappa.loc[i,'Min'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'Accuracy'].values.min()
#     
# 
# ax[1].set_ylabel('Harmonic Mean Class Accuracy',fontsize=10)
# #rotate tick labels 
# for tick in ax[1].get_yticklabels():
#     tick.set_rotation(0)
#     tick.set_fontsize(7)
# 
# 
# 
# #fig.tight_layout()
# 
# #ax.legend(fontsize=17,loc='lower center',bbox_to_anchor=(0.5, -0.47))
# 
# df_kappa.index.name = 'k_neighbours'
# 
# ax[1].get_xaxis().set_visible(False)
# 
# #fig.savefig('/Users/thomasdrayton/Desktop/univarite_rank.png',dpi=130)
# 
# #%% KAPPA univariate for different values of k
# 
# # Plot comparison of k for ranking
# #fig,ax = plt.subplots(1,1,figsize = (16,10))
# 
# # New df to calculate the mean Kappa statistic for Youden's J statistic
# df_kappa = pd.DataFrame(index=[7,5,3,1],columns = ['k-Neighbours','Max','Min']) 
# 
# for i in [7,5,3,1]:
#     ax[0].bar(x=df_metrics[df_metrics.k_neighbours==i].loc[:,'Variable'].values,
#                         height=df_metrics[df_metrics.k_neighbours==i].loc[:,'Kappa'],
#                         align='center',
#                         width=0.5,
#                         color='darkblue',
#                         label=r'$k$-neighbours:{0}'.format(i),
#                         alpha=i/10) #
#     
#     # rotate tick labels               
#     #for tick in ax.get_xticklabels():
#      #   tick.set_rotation(90)
#       #  tick.set_fontsize(18)
#     
#     # find total total Kappa Statisitc for each k_neighbour
#     df_kappa.loc[i,'k-Neighbours'] = i
#     df_kappa.loc[i,'Mean Cohen\'s Kappa'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'Kappa'].values.mean()
#     df_kappa.loc[i,'Max'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'Kappa'].values.max()
#     df_kappa.loc[i,'Min'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'Kappa'].values.min()
#     
# 
# ax[0].set_ylabel('Cohen\'s Kappa',fontsize=10)
# #rotate tick labels 
# for tick in ax[0].get_yticklabels():
#     tick.set_rotation(0)
#     tick.set_fontsize(7)
# 
# 
# ax[0].get_xaxis().set_visible(False)
# 
# 
# #fig.tight_layout()
# 
# #ax.legend(fontsize=20)
# 
# df_kappa.index.name = 'k_neighbours'
# 
# 
# #fig.savefig('/Users/thomasdrayton/Desktop/univarite_rank.png',dpi=100)
# 
# #%% MCC univariate for different values of k
# 
# # Plot comparison of k for ranking
# #fig,ax = plt.subplots(1,1,figsize = (16,10))
# 
# # New df to calculate the mean Kappa statistic for Youden's J statistic
# df_kappa = pd.DataFrame(index=[7,5,3,1],columns = ['k-Neighbours','Max','Min']) 
# 
# for i in [7,5,3,1]:
#     ax[2].bar(x=df_metrics[df_metrics.k_neighbours==i].loc[:,'Variable'].values,
#                         height=df_metrics[df_metrics.k_neighbours==i].loc[:,'MCC'],
#                         align='center',
#                         width=0.5,
#                         color='darkblue',
#                         label=r'$k$-neighbours:{0}'.format(i),
#                         alpha=i/10)
#     
#     # rotate tick labels               
#     for tick in ax[2].get_xticklabels():
#         tick.set_rotation(90)
#         tick.set_fontsize(9)
#     
#     # find total total Kappa Statisitc for each k_neighbour
#     df_kappa.loc[i,'k-Neighbours'] = i
#     df_kappa.loc[i,'Mean MCC'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'MCC'].values.mean()
#     df_kappa.loc[i,'Max'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'MCC'].values.max()
#     df_kappa.loc[i,'Min'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'MCC'].values.min()
#     
# 
# ax[2].set_ylabel('Matthews Correlation Coefficient',fontsize=10)
# #rotate tick labels 
# for tick in ax[2].get_yticklabels():
#     tick.set_rotation(0)
#     tick.set_fontsize(7)
# 
# 
# 
# fig.tight_layout()
# 
# ax[2].legend(fontsize=10, loc='lower center',bbox_to_anchor=(0.5, -0.75))
# 
# 
# df_kappa.index.name = 'k_neighbours'
# 
# #fig.savefig('/Users/thomasdrayton/Desktop/univarite_rank.png',dpi=300)
# 
# 
# 
# 
# =============================================================================
#%% SFS - requires univariate to be run first


# all variables to be tested
variables = ['tfuelv_f','toilv_f','LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
        'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
        'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
        'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

df_sfs = pd.DataFrame(columns = ['Variable Set','Accuracy','Kappa','Youden','TNR','TPR','Miss_Rate','FPR','k_neighbours','Variable'])

# and Cohen's Kappa
# starting with most discriminatory variable from univariate
sfs_list = [df_metrics_k1.loc[0,'Variable']]                                     # CHANGE K
variables.remove(sfs_list[0])
# labels for data - dont change
y = df_subset.loc[:,'Passed'].values


#Instantiate stratified kfold cross validation
k = 10
cv = StratifiedKFold(n_splits=k) 


# instantiate knn
nn = 1                                                                          # CHANGE
knn = KNeighborsClassifier(n_neighbors = nn,metric='cosine')

for i in range(len(variables)):
    
    #df to store variable scores [make fresh copy each iteration for later ranking]
    df_var_scores = pd.DataFrame(columns = ['Variable Set','Accuracy','Kappa','Youden','TNR','TPR','Miss_Rate','FPR','k_neighbours','Variable'])

    # going through variables to see which is the best each iteration in SFS
    count = 0
    for var in variables:
    
        
        # variables to test in current SFS iteration
        current_test_vars = []
        current_test_vars = sfs_list + [var]
        
        # input data
        X = df_subset.loc[:,current_test_vars].values
        
        kappa = []
        acc = []
        tnr = []
        tpr = []
        youden = []
        miss_rate = []
        fpr = []
        
        for train, test in cv.split(X, y):
            # Create current train and test data
            X_train = X[train,:]
            y_train = y[train]
            X_test = X[test,:]
            y_test = y[test]
            
            # Train kNN / store data point values and corresponding class from training set
            knn.fit(X_train,y_train)
            
            # Test
            y_pred = knn.predict(X_test)
            
            # Youden J statistic
            youden.append(balanced_accuracy_score(y_test,y_pred,adjusted=False))
            
            # Confusion matrix
            TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()
            total = TN+FP+FN+TP 
            
            # Cohen's Kappa Coefficient 
            exp = ((TN+FP)*(TN+FN) + (FN+TP)*(FP+TP))/(total*total)
            obs = (TP + TN) / (TP + FN + FP + TN)
            cohen_kappa = (obs - exp) / (1 - exp)
            kappa.append(cohen_kappa)
            
            acc.append(2/(((TP+FN)/TP)+((TN+FP)/TN)))
            
            # True negative rate / specificity & sensitivity
            tnr.append(TN/(TN+FP))      # assess faulty data
            tpr.append(TP/(TP+FN))   # using sensitivity because it's there to assess normal data
            miss_rate.append(FN/(FN+TP))
            fpr.append(FP/(FP+TN))
            
        # store the average scores
        df_var_scores.loc[count,"Variable"] = var
        df_var_scores.loc[count,"Variable Set"] = current_test_vars
        df_var_scores.loc[count,"Accuracy"] = np.mean(acc)
        df_var_scores.loc[count,'TNR'] = np.mean(tnr)
        df_var_scores.loc[count,'TPR'] = np.mean(tpr)
        df_var_scores.loc[count,'Miss_Rate'] = np.mean(miss_rate)
        df_var_scores.loc[count,'FPR'] = np.mean(fpr)
        df_var_scores.loc[count,"Kappa"] = np.mean(kappa)
        df_var_scores.loc[count,"k_neighbours"] = nn
        df_var_scores.loc[count,"Youden"] = np.mean(youden)

        
        count+=1
        
    # rank the average by kappa after all variables have been tested            
    df_var_scores.sort_values(by=sorting_v,inplace = True,ascending=False)
    df_var_scores.reset_index(inplace=True,drop=True)
    
    # add to sfs list
    sfs_list.append(df_var_scores.Variable.values[0])
    
    # store selected feature scores
    df_sfs = df_sfs.append(df_var_scores.loc[0,:])
    
    # remove selected feature from list so that it's not included in next iteration
    variables.remove(df_var_scores.Variable.values[0])
    
print(sfs_list)

# CHANGE YOURSELF : add univariate scores to data frame by looking at df_metrics_k1 or k3

# add univariate score
d = df_metrics_k1.loc[0,:].to_frame()                                           # CHANGE K
d = d.transpose()
df_sfs = d.append(df_sfs, ignore_index=True)


# shiftindex by 1 so starts at 1
df_sfs.index += 1


#%% Plot results

fig,ax = plt.subplots(1,2,figsize=(16,6),squeeze=False)                         # COMMENT OUT OR NOT

ax[0][0].plot(df_sfs.index,df_sfs.loc[:,sorting_v],label='Harmonic Mean Class Accuracy',color='k')   # CHANGE LABEL          Harmonic Mean Class Accruacy Kappa Statistic
ax[0][0].plot(df_sfs.index,df_sfs.TNR,label='TNR',color='k',linestyle='-.')                         # CHANGE INDEX
ax[0][0].plot(df_sfs.index,df_sfs.TPR,label='TPR',color='k',linestyle=':')
ax[0][0].plot(df_sfs.index,df_sfs.Miss_Rate,label='Miss Rate',color='r',linestyle=':')
ax[0][0].plot(df_sfs.index,df_sfs.FPR,label='FPR',color='blue',linestyle=':')

ax[0][0].set_ylabel('Performance',fontsize=19)
ax[0][1].set_ylabel('Performance',fontsize=19)
ax[0][0].set_xlabel('Number of Features',fontsize=19)
ax[0][1].set_xlabel('Number of Features',fontsize=19)
ax[0][0].set_title('k = 1',fontsize=19)
ax[0][1].set_title('k = 3',fontsize=19)
ax[0][0].tick_params('both',labelsize=13)
ax[0][1].tick_params('both',labelsize=13)


fig.text(0.52, 0.92, 'Hand-Labelled Performance Curve', ha='center',fontsize=21)   # CHANGE
#ax1 = ax.twinx()
#ax1.plot(df_sfs.index,df_sfs.Accuracy,label='Accuracy',color = 'blue')
#ax1.set_ylabel('Accuracy',color ='blue')
#ax1.tick_params(axis='y',labelcolor='blue')


# Shrink current axis's height by 10% on the bottom
box = ax[0][0].get_position()
ax[0][0].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
box = ax[0][1].get_position()
ax[0][1].set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax[0][0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.14),fancybox=True, ncol=5,fontsize=16)

ax[0][0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0][1].xaxis.set_major_locator(MaxNLocator(integer=True))


#%%

#fig.savefig('/Users/thomasdrayton/Desktop/FYP/Final Report/SFS/sfs_kappa_fl.png',format='png', dpi=200)

