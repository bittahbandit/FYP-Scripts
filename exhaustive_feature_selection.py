#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:46:08 2018

    - Exhaustive feature selection using Mxtend package
    
    - import df_subset i.e. hand labelled or fully labelled 
    
    - uses :
        - NN algorithm where k=3
        - balanced accuracy function i.e. Youden Statistic for 2 classes
        - 10 fold stratified cross validation
    
    - EXHAUSTUVE so takes 7-8hrs to run using all CPU cores
    
    - DONT RUN ANTHING ELSE WITH IT 
    
    - look at:
http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/
    
@author: thomasdrayton
"""

import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import accuracy_score


# all variables to use in exhaustive feature selection
variables = ['tfuelv_f','toilv_f','LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
        'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
        'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
        'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

# import dataset to use
X = df_subset.loc[:,variables].values
y = df_subset.Passed.values

# Use balanced accuracy to compensate for skew
#my_scorer = make_scorer(balanced_accuracy_score,adjusted=True)
my_scorer = make_scorer(cohen_kappa_score)

knn = KNeighborsClassifier(n_neighbors=3)

efs1 = EFS(knn, 
           min_features=1, 
           max_features=20,
           scoring=my_scorer,
           print_progress=True,
           cv=10,
           n_jobs=-1)   # use all CPU cores


efs1 = efs1.fit(X, y, custom_feature_names=variables)

df_efs = pd.DataFrame.from_dict(efs1.get_metric_dict()).T
df_efs.sort_values('avg_score', inplace=True, ascending=False)



os.system('say "your program has finished"')


df_efs.rename(columns={'avg_score': 'kappa_avg_score'}, inplace=True)
#%% print best subset

print('Best subset (corresponding names):', efs1.best_feature_names_)


#%% Save to csv
#df_efs.to_csv('/Users/thomasdrayton/Desktop/FYP/Code/workspace data/Whole performance curve labelled/df_efs_kapp_cv10_fl.csv')

#%% save to txt

#df_efs.to_csv('/Users/thomasdrayton/Desktop/FYP/Code/workspace data/Whole performance curve labelled/df_efs_kapp_cv10_fl.txt',
#          sep='\t',
#          index=False)




#%% Attmepting to make my own scorer function to use in feature selection 

#make youre own accuracy function and check it with the accuracy ones

def scorer(estimator, X, y):
    
    estimator.fit(X,y)
    y_pred = estimator.predict(X)
    
    print(y)
    print(confusion_matrix(y,y_pred))
    TN, FP, FN, TP = confusion_matrix(y,y_pred).ravel()
    
    return (TN+TP)/(TN+FP+TP+FN)
    
#%%
    
# =============================================================================
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import load_iris
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
# from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
# from sklearn.metrics import make_scorer
# 
# 
# 
# 
# 
# 
# 
# my_scorer = make_scorer(cohen_kappa_score)
# 
# iris = load_iris()
# X = iris.data
# y = iris.target
# 
# knn = KNeighborsClassifier(n_neighbors=3)
# 
# efs1 = EFS(knn, 
#            min_features=1,
#            max_features=4,
#            scoring=my_scorer,
#            print_progress=True,
#            cv=20)
# 
# efs1 = efs1.fit(X, y)
# 
# df = pd.DataFrame.from_dict(efs1.get_metric_dict()).T
# df.sort_values('avg_score', inplace=True, ascending=False)
# 
# print(df)
# 
# =============================================================================

df_efs_t10_hl = df_efs.head(10)

df_efs_t10_hl.to_excel('/Users/thomasdrayton/Desktop/FYP/Code/workspace data/Hand Labelled performance curve/df_efs_kapp_cv10_t10_hl.xlsx')
#%%
