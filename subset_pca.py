#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:41:16 2018

IMPORT SUBSET BEFORE YOUR RUN THE CODE

@author: thomasdrayton
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# INITIALISE
# dataframe to use: assigned from imported data
df = df_subset000

#%% save labelled data and drop Passed column in preparation for dim. reduction
passed = df.Passed


#%% Select ALL variables for dimensionality reduction

cols = ['n1v_f','tfuelv_f','toilv_f','LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
        'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
        'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
        'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

# without n1v
cols = ['tfuelv_f','toilv_f','LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
        'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
        'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
        'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']

#%% Select a subset of variables

#fig,ax = plt.subplots(2,1,figsize = (6.8,7),squeeze=False) #full page length (6.8, 30) (6.2, 30)  (6.8,6.8)         # CHANGE:  comment out


# selecting subplot
sp = 1
sp2 = 0

#ax[sp][sp2].set_xlim(-1,1)
#ax[sp][sp2].set_ylim(-0.6,1)                                                           # CHANGE:

#cols = ['toilv_f', 't30v_f', 'p20v_f', 'ps26v_f', 'tgtv_f', 'poilv_S_f']

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
cols = ['IP_Vib_Magnitude_A_f', 't20v_f', 't25v_f', 't30v_f', 'p30v_f', 'P44_f', 'trav_f', 'tsasv_S_f']

# instantiante PCA object - reduction to 2 principle components
pca = PCA(n_components=2)

# create data that PCA will be performed on 
f_vals = df.loc[:,cols].values


#%%
# Perform PCA on data set
principalComponents = pca.fit_transform(f_vals)

# Find the explained variance for each principle component
pc_variance = pca.explained_variance_ratio_

# Create a new dataframe with each column being a princple component
pcDf = pd.DataFrame(data = principalComponents, columns = ['pc1','pc2'])

#%% Label data

# =============================================================================
# 
# #finalDf = pd.concat([Passed,pcDf], axis = 1)
# faulty_P_Curve = [('T081015',3.0),('T154058',2.0),('T154058',3.0),('T102342',2.0),('T012939',3.0),('T001707',3.0),('T102356',2.0),('T102356',3.0),('T012648',2.0),('T012648',3.0),('T152228',2.0),('T152228',3.0),('T201046',3.0),('T011013',3.0),('T082208',3.0),('T103720',3.0),('T103720',5.0),('T013704',3.0),('T124744',3.0),('T123222',3.0),('T013339',3.0),('T150349',3.0),('T000735',3.0),('T083638',3.0),('T205540',3.0),('T083331',3.0),('T082137',0.0)]
# faulty_pcurve_annotations = []
# for i in df.index:
#     for ii in faulty_P_Curve:
#         if(i == ii):
#             faulty_pcurve_annotations.append(1)
#             break
#     if(i!=ii):
#         faulty_pcurve_annotations.append(0)
#     
# print(len(df_subset.index))
# 
# =============================================================================

# Concat Passed column by reseting index for both
passed.reset_index(drop=True,inplace = True)
pcDf = pd.concat([pcDf,passed],axis = 1)

#%%
# Separate into faulty and non faulty
norm = pcDf.loc[pcDf['Passed'] == 1]
fnorm = pcDf.loc[pcDf['Passed'] == 0]

#%% Plot PCA



fig.subplots_adjust(top=0.95, wspace=0.3, hspace=0.3,bottom=0.09)

# plot normal and faulty
ax[sp][sp2].scatter(norm.loc[:,'pc1'],norm.loc[:,'pc2'], s=1)
ax[sp][sp2].scatter(fnorm.loc[:,'pc1'],fnorm.loc[:,'pc2'],marker = 'x', s=30,c='red',
           linewidth=0.7)
ax[sp][sp2].set_xlabel('PC1 ({:.2f}% explained var.)'.format(100*pc_variance[0]), fontsize = 9)
ax[sp][sp2].set_ylabel('PC2 ({:.2f}% exlpained var.)'.format(100*pc_variance[1]), fontsize = 9)
fig.text(0.52, 0.47, 'Optimal EFS Subset: Hand-Labelled Performance Curve', ha='center',fontsize=10)               # CHANGE 0.3  0.73
#ax.set_ylim(-1.2,0.605)
#ax.set_xlim(-0.8,1)


ax[sp][sp2].legend(['Normal','Faulty'],fontsize = 9)


# number of original dimensions: 6
# =============================================================================
# coeff = np.transpose(pca.components_)
# n = coeff.shape[0]
# # Loading plot
# for i in range(n):
#     ax[sp][sp2].arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)   # plotting the arrow based pca.components: each pair of values (after transposing pca.components) is (pc1 correlation, pc2 correlation) for each original feature dimension   
#     ax[sp][sp2].text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, cols[i], color = 'g', ha = 'center', va = 'center') # adding the specified labels of the orginal dimensions to the arrows
# =============================================================================


#%%
#Adding all test file labels to plots
#for i, txt in enumerate(df.index):
#    ax.annotate(txt, (pcDf.iloc[i,0], pcDf.iloc[i,1]),fontsize = 3)


#%%   function for plotting Bi-plot / loading plot

def myplot(score,coeff,labels=None):
    xs = score[:,0]     # 1st principal component
    ys = score[:,1]     # 2nd pc
    n = coeff.shape[0]   # gettting the original number of dimensions 
    scalex = 1.0/(xs.max() - xs.min())  # scaling pc1 to between 1 and -1
    scaley = 1.0/(ys.max() - ys.min())  # scaling pc2 to between 1 and -1
    plt.scatter(xs * scalex,ys * scaley, c = y)   # plotting pc1 vs pc2
    for i in range(n):   # for the length of the orginal dimension
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)   # plotting the arrow based pca.components: each pair of values (after transposing pca.components) is (pc1 correlation, pc2 correlation) for each original feature dimension   
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center') # adding the specified labels of the orginal dimensions to the arrows  
#plt.xlim(-1,1)
#plt.ylim(-1,1)
#plt.xlabel("PC{}".format(1))
#plt.ylabel("PC{}".format(2))
#plt.grid()

#Call the function. Use only the 2 PCs.
#myplot(x_new[:,0:2],np.transpose(pca.components_[0:2, :]))
#plt.show()

#%%

#fig.savefig('/Users/thomasdrayton/Desktop/PCA_efs_fl_hl.png',dpi=400,format='png')
#%%
#ax[sp][sp2].legend(['Normal','Faulty'],loc = 4,fontsize = 9)
ax[sp][sp2].legend(['Normal','Faulty'],fontsize = 9,loc=4)
