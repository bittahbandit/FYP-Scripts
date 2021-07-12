#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:46:01 2019

    - univariate ranking of feature selection w/ bar charts of performance
    
    - cross validation is used for training and testing
    
    - Performance for Accuracy, Kappa Statistic and Specificity are used
    
    - import df_subset_all_f_win.spydata  or  df_subset.spydata
    
    - Followed by sequential feature selection


@author: thomasdrayton
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from matplotlib.ticker import MaxNLocator
#%%  plot showing predicted probabilities

def plot_predict_proba():
    h =  0.006 # mesh step size
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    #Z = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig,ax = plt.subplots()
    
    cmap_bright = ListedColormap(['#FF0000', '#0000FF'])
    cm = plt.cm.RdBu
    #ax.pcolormesh(xx, yy, Z, cmap=cmap_bright,alpha=.6)
    
    # Put the result into a color plot
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
    # Plot also the training points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bright,
               edgecolor='k', s=20)
    
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bright,
               edgecolors='k',alpha = 0.6)
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bright,
               edgecolors='k')



# Change to latex font ------------------------------------------------
plt.rcParams["font.family"] = "serif"


variables = ['tfuelv_f','toilv_f','LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
        'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
        'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
        'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

# Dataframe to store metric
df_metrics = pd.DataFrame(columns = ['Variable','Accuracy','Kappa','Specificity','k_neighbours'])

i_ft = 0
# iterating through different k values in knn
for nn in [1,3,5,7]:
    # Instantiate kNN classifier
    knn = KNeighborsClassifier(n_neighbors = nn)
    
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
            
            # Calc accuracy from fold test and store result
            acc.append(accuracy_score(y_test,y_pred))
            
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
             
            # True negative rate / specificity
            tnr.append(TN/(TN+FP))
        
    
        df_metrics.loc[i_ft,"k_neighbours"] = nn
        df_metrics.loc[i_ft,"Variable"] = ft
        df_metrics.loc[i_ft,"Accuracy"] = np.mean(acc)
        df_metrics.loc[i_ft,"Kappa"] = np.mean(kappa)
        df_metrics.loc[i_ft,'Specificity'] = np.mean(tnr)
        
    
    
        i_ft+=1
    




#%% Plot all bar chart of Kappa statistic for all variables k_neighbours 1,3,5,7
# Plot comparison of k for ranking
fig,ax = plt.subplots(1,1,figsize = (16,10))

df_kappa = pd.DataFrame(index=[7,5,3,1],columns = ['Mean Kappa','Max','Min'])

for i in [7,5,3,1]:
    ax.bar(x=df_metrics[df_metrics.k_neighbours==i].loc[:,'Variable'].values,
                        height=df_metrics[df_metrics.k_neighbours==i].loc[:,'Kappa'],
                        align='center',
                        width=0.5,
                        color='darkblue',
                        label='k_neighbours:{0}'.format(i),
                        alpha=i/10)
    
    # rotate tick labels               
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
    
    # find total total Kappa Statisitc for each k_neighbour
    df_kappa.loc[i,'Mean Kappa'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'Kappa'].values.mean()
    df_kappa.loc[i,'Max'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'Kappa'].values.max()
    df_kappa.loc[i,'Min'] = df_metrics[df_metrics.k_neighbours==i].loc[:,'Kappa'].values.min()
    
    
ax.set_ylabel('Kappa Statistic')

fig.tight_layout()

ax.legend()

df_kappa.index.name = 'k_neighbours'


#%% Order variables based on their classification performance

# Order k=1 based on kappa statistic
df_metrics_k1 = df_metrics[df_metrics.k_neighbours==1]                          # df for metric k1
df_metrics_k1.sort_values(by='Kappa',inplace = True,ascending=False)

# plot results
fig,ax = plt.subplots(1,1,figsize = (16,10))
ax.bar(x=df_metrics_k1.loc[:,'Variable'].values,
       height=df_metrics_k1.loc[:,'Kappa'],
       align='center',
       width=0.5,
       color='darkblue',
       label='k_neighbours:{0}'.format(1),
       alpha=3/10)

# rotate tick labels               
for tick in ax.get_xticklabels():
    tick.set_rotation(60)

ax.set_ylabel('Kappa Statistic')

ax.legend()
fig.tight_layout()

#%% Sequential Feature Selection for form 1-NN results

cols = ['Variables','Accuracy','Kappa','Specificity']
df_sfs1 = pd.DataFrame(columns=cols)

# output labels - same for each feature value
y = df_subset.loc[:,'Passed'].values

# iterate and slice rank 1 by 1
odr_var = []
i_rnk = 1
for v in df_metrics_k1.Variable:
    
    odr_var.append(v)
    
    if(i_rnk == 1):
        i_rnk+=1
        continue
    
    # obtain performance metric from stratified cv train test 
    # Prepare data X,y
    X = df_subset.loc[:,odr_var].values    
    
    
    # lists to store metrics for each fold to eventually average
    acc = []
    kappa = []
    tnr = []

    for train, test in cv.split(X, y):
        #print(train,test)
        
        # Create current train and test data
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        
        # Train kNN / store data point values and corresponding class from training set
        knn.fit(X_train,y_train)
        
        # Test
        y_pred = knn.predict(X_test)
        
        # Calc accuracy from fold test and store result
        acc.append(accuracy_score(y_test,y_pred))
        
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
             
        # True negative rate / specificity
        tnr.append(TN/(TN+FP))

    
    # store in new datat frame
    df_sfs1.loc[i_rnk,"Variables"] = df_subset.loc[:,odr_var].columns.values
    df_sfs1.loc[i_rnk,"Accuracy"] = np.mean(acc)
    df_sfs1.loc[i_rnk,"Kappa"] = np.mean(kappa)
    df_sfs1.loc[i_rnk,'Specificity'] = np.mean(tnr)
    
    
    i_rnk+=1


# plot results from 1NN sequential feature selection

fig,ax = plt.subplots(1,1,figsize = (16,10))

ax.plot(df_sfs.index,df_sfs.Kappa,label='Kappa Statistic',color='k')
ax.plot(df_sfs.index,df_sfs.Specificity,label='Specificity',color='k',
        linestyle='dashed')
ax.set_ylabel('Kappa and Specificity')
ax.set_xlabel('Ranked variables based on 1-NN')

ax1 = ax.twinx()
ax1.plot(df_sfs.index,df_sfs.Accuracy,label='Accuracy',color = 'blue')
ax1.set_ylabel('Accuracy',color ='blue')
ax1.tick_params(axis='y',labelcolor='blue')


ax.legend(loc='lower right')

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#%% Sequential Feature Selection for form 3-NN results

# Order k=3 based on kappa statistic
df_metrics_k3 = df_metrics[df_metrics.k_neighbours==3]                          # df for metric k1
df_metrics_k3.sort_values(by='Kappa',inplace = True,ascending=False)


cols = ['Variables','Accuracy','Kappa','Specificity']
df_sfs3 = pd.DataFrame(columns=cols)



# output labels - same for each feature value
y = df_subset.loc[:,'Passed'].values

# iterate and slice rank 1 by 1
odr_var = []
i_rnk = 1
for v in df_metrics_k3.Variable:
    
    odr_var.append(v)
    
    if(i_rnk == 1):
        i_rnk+=1
        continue
    
    # obtain performance metric from stratified cv train test 
    # Prepare data X,y
    X = df_subset.loc[:,odr_var].values    
    
    
    # lists to store metrics for each fold to eventually average
    acc = []
    kappa = []
    tnr = []

    for train, test in cv.split(X, y):
        #print(train,test)
        
        # Create current train and test data
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        
        # Train kNN / store data point values and corresponding class from training set
        knn.fit(X_train,y_train)
        
        # Test
        y_pred = knn.predict(X_test)
        
        # Calc accuracy from fold test and store result
        acc.append(accuracy_score(y_test,y_pred))
        
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
             
        # True negative rate / specificity
        tnr.append(TN/(TN+FP))

    
    # store in new datat frame
    df_sfs3.loc[i_rnk,"Variables"] = df_subset.loc[:,odr_var].columns.values
    df_sfs3.loc[i_rnk,"Accuracy"] = np.mean(acc)
    df_sfs3.loc[i_rnk,"Kappa"] = np.mean(kappa)
    df_sfs3.loc[i_rnk,'Specificity'] = np.mean(tnr)
    
    
    i_rnk+=1

# ---------------------------------------------------------------------------
# plot results from 3NN sequential feature selection
    
fig,ax = plt.subplots(1,1,figsize = (16,10))

ax.plot(df_sfs3.index,df_sfs3.Kappa,label='Kappa Statistic',color='k')
ax.plot(df_sfs3.index,df_sfs3.Specificity,label='Specificity',color='k',
        linestyle='dashed')
ax.set_ylabel('Kappa and Specificity')
ax.set_xlabel('Ranked variables based on 3-NN')

ax1 = ax.twinx()
ax1.plot(df_sfs3.index,df_sfs3.Accuracy,label='Accuracy',color = 'blue')
ax1.set_ylabel('Accuracy',color ='blue')
ax1.tick_params(axis='y',labelcolor='blue')


ax.legend(loc='lower right')

ax.xaxis.set_major_locator(MaxNLocator(integer=True))


#%% Sequential Feature Selection for form 5-NN results

# Order k=3 based on kappa statistic
df_metrics_k5 = df_metrics[df_metrics.k_neighbours==5]                          # df for metric k1
df_metrics_k5.sort_values(by='Kappa',inplace = True,ascending=False)


cols = ['Variables','Accuracy','Kappa','Specificity']
df_sfs5 = pd.DataFrame(columns=cols)



# output labels - same for each feature value
y = df_subset.loc[:,'Passed'].values

# iterate and slice rank 1 by 1
odr_var = []
i_rnk = 1
for v in df_metrics_k5.Variable:
    
    odr_var.append(v)
    
    if(i_rnk == 1):
        i_rnk+=1
        continue
    
    # obtain performance metric from stratified cv train test 
    # Prepare data X,y
    X = df_subset.loc[:,odr_var].values    
    
    
    # lists to store metrics for each fold to eventually average
    acc = []
    kappa = []
    tnr = []

    for train, test in cv.split(X, y):
        #print(train,test)
        
        # Create current train and test data
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        
        # Train kNN / store data point values and corresponding class from training set
        knn.fit(X_train,y_train)
        
        # Test
        y_pred = knn.predict(X_test)
        
        # Calc accuracy from fold test and store result
        acc.append(accuracy_score(y_test,y_pred))
        
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
             
        # True negative rate / specificity
        tnr.append(TN/(TN+FP))

    
    # store in new datat frame
    df_sfs5.loc[i_rnk,"Variables"] = df_subset.loc[:,odr_var].columns.values
    df_sfs5.loc[i_rnk,"Accuracy"] = np.mean(acc)
    df_sfs5.loc[i_rnk,"Kappa"] = np.mean(kappa)
    df_sfs5.loc[i_rnk,'Specificity'] = np.mean(tnr)
    
    
    i_rnk+=1

# ---------------------------------------------------------------------------
# plot results from 3NN sequential feature selection
    
fig,ax = plt.subplots(1,1,figsize = (16,10))

ax.plot(df_sfs5.index,df_sfs5.Kappa,label='Kappa Statistic',color='k')
ax.plot(df_sfs5.index,df_sfs5.Specificity,label='Specificity',color='k',
        linestyle='dashed')
ax.set_ylabel('Kappa and Specificity')
ax.set_xlabel('Ranked variables based on 5-NN')

ax1 = ax.twinx()
ax1.plot(df_sfs5.index,df_sfs5.Accuracy,label='Accuracy',color = 'blue')
ax1.set_ylabel('Accuracy',color ='blue')
ax1.tick_params(axis='y',labelcolor='blue')


ax.legend(loc='lower right')

ax.xaxis.set_major_locator(MaxNLocator(integer=True))