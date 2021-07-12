#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:52:35 2019

    - Relative Risk v2
    
    - without evaluation functions so it's easier to debug
    
    - Need to run FishersLinearDiscriminant to get the normal and faulty data
    
    
    
    - 

@author: thomasdrayton
"""




import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold
from matplotlib import cm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.path as mpltPath
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import itertools
from operator import itemgetter
import scipy.stats as st
from scipy import interp
from sklearn.metrics import auc
from matplotlib import animation

#%%

# Functions -----------------------------------------------------------------


def findKDEbandwidth(bandwidths , fitting_data, class_type):
    '''
    Bandwidth hyperparameter gridsearch so that the probability density 
    function fits the data well. Uses 5 fold cross validation.
    
    Bandwidths is the array of values to try i.e. np.linspace()
    
    class_type is a string: either 'Faulty' or 'Normal'
    '''
    gridS = GridSearchCV(KernelDensity(kernel='gaussian'),
                         {'bandwidth': bandwidths},
                         cv=KFold(5),
                         iid=False,
                         n_jobs=-1)
    gridS.fit(fitting_data)
    
    print(class_type,': ',gridS.best_params_)
    return gridS
    




def kde2D(x, y, bandwidth, x_mesh=[0,1,100j],y_mesh=[0,1,100j], **kwargs): 
    """
    Build 2D kernel density estimate (KDE).
    https://tinyurl.com/y6goqsve
    """

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x_mesh[0]:x_mesh[1]:x_mesh[2], 
                      y_mesh[0]:y_mesh[1]:y_mesh[2]]                                    # 100j => 100 points linearly spaced [number of bins]]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    # Instantiate Kernel 
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    #z = kde_skl.score_samples(xy_sample)
    
    #positions = np.vstack([xx.ravel(), yy.ravel()])
    #values = np.vstack([x, y])
    #kernel = st.gaussian_kde(values)
    #z = np.reshape(kernel(positions).T, xx.shape)
    
    return xx, yy, np.reshape(z, xx.shape)

    




def levelKDE(surface):
    '''
    Anything below 10^(-5) is 10^(-5)
    
    surface is zz_f or zz_n from Kde2D
    '''
    for i,vals in enumerate(surface):
        for ii,val in enumerate(vals):
            if(val <= 10**(-5)):
                surface[i][ii] = 10**(-5)
    return surface



def encapsulatingContours(set_of_contours,level):
    '''
    set_of_contours = contSet[lvl]  which contSet is the variable that stores
                                    all the levels where each lvl contains the
                                    number of contours for that level
     level = lvl                              
    '''
    cont_num = range(len(set_of_contours[level])) # number of contours            -----------
    conts_inside = []       # list for contours within each other       This can be a function that takes contSet[lvl] as an input
    for c in itertools.combinations(cont_num,2):
        # check if contours are inside each other
        contour1 = set_of_contours[level][c[0]]
        contour2 = set_of_contours[level][c[1]]
        #print(c)
        path1 = mpltPath.Path(contour1)
        path2 = mpltPath.Path(contour2)
        
        if(path1.contains_path(path2)):
            # path2 is inside path1
            conts_inside.append(c)
            
        elif(path2.contains_path(path1)):
            # path1 is inside path2                                     and returns conts_inside
            conts_inside.append(c)
    return conts_inside    




def contourPlot(xx,yy,surface,fold_count,data_name,data1=None,data2=None):
    '''
    data that's been used to create KDE'
    data_name is a string of the data that you used 
    '''
    fig,ax = plt.subplots(1,1)
    cf = ax.contourf(xx,yy,surface,levels=15,cmap=cm.plasma)
    if(data1.all()!=None):
        ax.scatter(data1[:,0], data1[:,1], marker = 'x', s=70,c='r',linewidth=0.7)
        ax.scatter(data2[:,0], data2[:,1], marker = 'o', s=20,edgecolor='white',linewidth=0.7,facecolor='None')
    fig.colorbar(cf)
    ax.set_title("Faulty/Normal Log-likelihood: Fold {0}".format(fold_count+1))
    
    
#contourPlot(xx,yy,log_rr,fold_count,'',data1=f_X_train,data2=n_X_train)     

                                  # make it so that all plots are shown in a single figure


def surfacePlot(xx,yy,surface,fold_count):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    cb = ax.plot_surface(xx,yy,surface,antialiased=False,
                 cmap=cm.plasma,
                 alpha=1)
    ax.set_title("Faulty/Normal Log-likelihood: Fold {0}".format(fold_count+1))
    #fig.colorbar(cb)
    
    for angle in np.linspace(0, 360,600):
        ax.view_init(30, angle)
        #plt.draw()
        plt.pause(.0001)




    
    
#%%


def plotDB( p, n, f, l,pause_time=5):
    '''
    This will give the actual results for each contour that is plotted
    Use this to varify that you're automated results are correct'
    
    -   p is the Nx2 array containing the coordinates of the contour to plot
        p = contSet[lvl][contour]
        
    -   n is the normal data
    
    -   f is the faulty data
    
    -   if l == '+' positive levels, l=='-' => negative levels
    '''
    
    
    for i in p:
        
        # create paths ------------------------------------------------
        path = mpltPath.Path(i)
        
        n_in = path.contains_points(n)
        f_in = path.contains_points(f)
        
        # assign points ------------------------------------------------
        
        if(l=='+'):
            tn_fn_colors = []
            for point in f_in:                                                  # for each faulty point that's inside contour
                if(point==True):                                                    # if faulty point is inside contour
                    tn_fn_colors.append('red')                                          # give it colour RED  (TN)
                else:                                                               # if faulty point is outside contour
                    tn_fn_colors.append('blue')                                         # FP (becuase region outside contour is positive/normal)
                
            tp_fp_colours = []
            for point in n_in:                                                  # for each normal point that's inside contour
                if(point==True):                                                    # if faulty point is inside contour
                    tp_fp_colours.append('orange')                                      # normal point inside is FN
                else:                                                               # if faulty point is otuside contour        
                    tp_fp_colours.append('purple')                                      # normal points outside is TP
            print('TN: ',np.sum(f_in))
            print('FN: ',np.sum(n_in))
            print()
            
                                                                                # for negative regions where contours define normal 
        if(l=='-'):
            tn_fn_colors = []                       
            for point in f_in:                                                  # for each fautly point that's inside contour
                if(point==True):
                    tn_fn_colors.append('red') # faulty point in contour is FP
                else:
                    tn_fn_colors.append('blue') # faulty point outside contour is TN
                    
            tp_fp_colours = []
            for point in n_in:
                if(point==True):
                    tp_fp_colours.append('orange') # normal point inside is TP
                else:
                    tp_fp_colours.append('purple') # normal points outside is FN
            print('FP: ',np.sum(f_in))
            print('TP: ',np.sum(n_in))
            print()
                    
        # plot DB -------------------------------------------------------
        plt.plot(i[:,0],i[:,1],c='k') # path
        
        plt.scatter(f[:,0],f[:,1],marker='x',s=50,color=tn_fn_colors) # faulty points
        plt.scatter(n[:,0],n[:,1],marker='^',s=30,facecolor=None,edgecolors=tp_fp_colours) # normal points
        
        
        
        
        plt.show()
        plt.pause(pause_time)

       


def pointInOrOut(data,set_of_contours,level_idx,in_colour,out_colour):
    '''
    Given datapoints and Contour set with associated level, 
    
    Returns boolean array of data points are inside or outside contours
    
    data is np.array [Nx2]
    
    '''
    inside = []
    for i in data: # for each index in faulty data
        
        flag = 0
        
        conts_inside = encapsulatingContours(set_of_contours,level_idx)
        
        # ----------------------------------------------------
        # get boolean array for contours that contain the point
        
        # contours that point is within
        inside_lst = []
        
        # for each point, check if it is inside any of the contours
        for contour_coords in set_of_contours[level_idx]:
            
            path = mpltPath.Path(contour_coords)
            
            # check whether in or out of current contour
            f_in = path.contains_point(i)
            
            # add f_in result to list for comparison later
            if(f_in==True):
                inside_lst.append(in_colour)
            else:
                inside_lst.append(out_colour)
                
            #inside_lst.append(f_in)
        
        # ----------------------------------------------------
        
        # ----------------------------------------------------       
        
        # if f_in_lst is all False => not in a contour
        if(all(i == out_colour for i in inside_lst)):
            inside.append(out_colour)
            flag = 1
            continue # move onto next iteration
            
        # ----------------------------------------------------
        
        # ---------------------------------------------------
        # check if it's inside a contour of another contour => False
        # compare list to see if it's within list - if point is true for same index as second value in tuple in list of ti
        for i in conts_inside:
            for ii in range(len(inside_lst)): # if they are in at the same position
                if((i[1]==ii) and (inside_lst[ii]==in_colour)):
                    inside.append(out_colour)
                    flag = 1
        # ---------------------------------------------------
    
        # ---------------------------------------------------
        # check to see if it's inside a big contour but not a little one
        if(flag==0):
            inside.append(in_colour)
        # ---------------------------------------------------
        
    return inside



def plotCleanDB(set_of_contours, level_idx, faulty_data, normal_data, pos_or_neg,r,c):
    
    '''
    usage: plotCleanDB(contSet,1,f_X_test,n_X_test,'+',0,0)
    '''
    
    fig, ax = plt.subplots(1,2,squeeze=False,figsize=(13,5.5))
    
    # plot all db for the lvl
    for i in set_of_contours[level_idx]:
        ax[r][c].scatter(i[:,0],i[:,1],c='k',s=0.2)
        
    if(pos_or_neg=='+'):                                                        # +ve
        
        #faulty and normaal boolean arrays for the data
        faulty_clrs = pointInOrOut(faulty_data,set_of_contours,level_idx,'red','blue')
        normal_clrs = pointInOrOut(normal_data,set_of_contours,level_idx,'orange','purple')
        
        # plot color coded data
        ax[r][c].scatter(faulty_data[:,0], faulty_data[:,1], marker = 'x', s=50,
                    c=faulty_clrs,
                    linewidth=0.7)
        ax[r][c].scatter(normal_data[:,0], normal_data[:,1],marker = '^',
                    facecolor="None",
                    s=20,edgecolors= normal_clrs,
                    linewidths=0.5)
        

        
    else:                                                                       # -ve
        #faulty and normaal boolean arrays for the data
        faulty_clrs = pointInOrOut(faulty_data,set_of_contours,level_idx,'blue','red')      # colors switched
        normal_clrs = pointInOrOut(normal_data,set_of_contours,level_idx,'purple','orange')
        
        # plot color coded data
        ax[r][c].scatter(faulty_data[:,0], faulty_data[:,1], marker = 'x', s=50,
                    c=faulty_clrs,
                    linewidth=0.7)
        ax[r][c].scatter(normal_data[:,0], normal_data[:,1],marker = '^',
                    facecolor="None",
                    s=20,edgecolors= normal_clrs,
                    linewidths=0.5)
        

    
    # legend
    colors = ['red', 'blue']
    markers = [plt.scatter([], [], color=c, linewidth=0.8, marker='x') for c in colors]
    labels = ['TN: {0}'.format(faulty_clrs.count('red')),'FN: {0}'.format(faulty_clrs.count('blue'))]
    
    colors = ['purple', 'orange']
    [markers.append(plt.scatter([],[],marker='^',facecolor='None',s=25,edgecolor=c,linewidths=0.6)) for c in colors]
    labels.append('TP: {0}'.format(normal_clrs.count('purple')))
    labels.append('FP: {0}'.format(normal_clrs.count('orange')))
    
    ax[r][c].legend(markers,labels)
    
    # adjust subplot
    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.8, wspace=None, hspace=None)
    
    #fig.savefig('/Users/thomasdrayton/Desktop/neg_threshold_db.png',format='png',dpi=250)
    
#plotCleanDB(contSet,1,f_X_test,n_X_test,'+',0,0)
#plotCleanDB(contSet,99,f_X_test,n_X_test,'-',0,1)
    
        
def plotAverageROC(list_of_folds):
    '''
    list_of_folds = df_folds
    
    e.g. plotAverageRPC(df_folds)
    
    '''
    all_tpr = []
    all_fpr = []
    
    # create new dataframe from df_folds
    for fold in df_folds:
        
        # extract FPR  and TPR as array and add 
        all_tpr.append(fold.TPR)
        all_fpr.append(fold.FPR)
    
    all_tpr = np.transpose(np.array(all_tpr,dtype=float))
    all_fpr = np.transpose(np.array(all_fpr,dtype=float))
    
    avg_tpr = []
    avg_fpr = []
    # calculate averages
    for i in range(all_tpr.shape[0]):
        avg_tpr.append(np.nanmean(all_tpr[i,:]))
        avg_fpr.append(np.nanmean(all_fpr[i,:]))
        
    # plot ROC
    plt.plot(avg_fpr, avg_tpr,c='k',s = 1)
    plt.plot(np.linspace(0,1),np.linspace(0,1),c='lightgrey')
    plt.title('ROC curve from {0} fold stratified cross validation'.format(len(df_folds)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')







def plotAllFoldsROC(list_of_folds):
    
    fig, ax = plt.subplots()
    
    # calc average tpr and fpr from all folds --------------------
    all_tpr = []
    all_fpr = []
    aucs = []
    # create new dataframe from df_folds
    for fold in df_folds:
        
        # extract FPR  and TPR as array and add 
        all_tpr.append(np.array(fold.loc[:,'TPR'],dtype=float))
        all_fpr.append(np.array(fold.loc[:,'FPR'],dtype=float))
# =============================================================================
#         all_tpr.append(interp(np.linspace(0,1,400), np.array(fold.loc[:,'FPR'],dtype=float), np.array(fold.loc[:,'TPR'],dtype=float)))
#         all_tpr[-1][0] = 0.0
#         roc_auc = auc(np.array(fold.loc[:,'FPR'],dtype=float), np.array(fold.loc[:,'TPR'],dtype=float))
#         aucs.append(roc_auc)
#         
#     mean_tpr = np.mean(all_tpr, axis=0)
#     mean_tpr[-1] = 1.0 # ignore
#     mean_auc = auc(np.linspace(0,1,400), mean_tpr)
#     std_auc = np.std(aucs)
#     plt.plot(np.linspace(0,1,400), mean_tpr, color='b',
#              label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)
#     
#     std_tpr = np.std(all_tpr, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(np.linspace(0,1,400), tprs_lower, tprs_upper, color='grey', alpha=.2,
#                      label=r'$\pm$ 1 std. dev.')
# =============================================================================
    
    
    #all_tpr = np.array(fold.loc[:,'TPR'],dtype=float)
    #all_fpr = np.array(fold.loc[:,'FPR'],dtype=float)
    
    avg_tpr = np.mean(all_tpr, axis=0)
    avg_fpr = np.mean(all_fpr, axis=0)
    
    

    #avg_tpr = []
    #avg_fpr = []
    # calculate averages 
    #for i in range(all_tpr.shape[0]):
    #    avg_tpr.append(np.nanmean(all_tpr[i,:]))
    #    avg_fpr.append(np.nanmean(all_fpr[i,:]))

    #plot average
    ax.plot(np.sort(avg_fpr), np.sort(avg_tpr),c='k',lw=1.6,label='Mean: AUC={0:.2f}'.format(auc(avg_fpr[61:],avg_tpr[61:],reorder=True)))
    
    std_tpr = np.std(all_tpr, axis=0)
    std_fpr = np.std(all_fpr, axis=0)
    
    #tprs_upper = avg_tpr + std_tpr
    #fprs_upper = avg_fpr + std_fpr

    #tprs_lower = avg_tpr - std_tpr
    #fprs_lower = avg_fpr - std_fpr
    
    #plt.plot(fprs_lower,tprs_lower,ls='--')
    #plt.plot(fprs_upper,tprs_upper,ls='--')

    #plt.fill_between(avg_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
     #            label=r'$\pm$ 1 std. dev.')
    # std deviation for TPR --------------------
    #print(all_tpr.shape)
    #std_tprs = []
    #std_fprs = []
    #for r in range(all_tpr.shape[0]):
    #    std_tprs.append(np.std(all_tpr[r,:]))
    #    std_fprs.append(np.std(all_fpr[r,:]))
        
    
    # std dev bounds -----------------------
    #upper_t = np.array(avg_tpr) + np.array(std_tprs) 
    #lower_t = np.array(avg_tpr) - np.array(std_tprs) 
    #upper_f = np.array(avg_fpr) + np.array(std_fprs)
    #lower_f = np.array(avg_fpr) - np.array(std_fprs)
    
    
    
    
    # plot bounds
    #plt.fill_between(avg_fpr, lower_t, upper_t, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    #plt.fill_between(avg_fpr, lower_f, upper_f, color='grey', alpha=.2)
    
    
    
    # plot each fold
    for i,fold in enumerate(list_of_folds):
        roc_auc = auc(np.array(fold.loc[:,'FPR'],dtype=float), np.array(fold.loc[:,'TPR'],dtype=float),reorder=True)
        ax.plot(np.sort(fold.FPR),np.sort(fold.TPR),lw=0.4,label = 'Fold {1}: AUC={0:.2f}'.format(roc_auc,i+1),ls='-')
        
    
    ax.plot(np.linspace(0,1),np.linspace(0,1),c='lightgrey')
    ax.set_xlabel("FPR",fontsize=11)
    ax.set_ylabel("TPR",fontsize=11)
    ax.legend(fontsize=11)
    fig.savefig('/Users/thomasdrayton/Desktop/roc_complete_fld_cv10.png',format='png',dpi=400)
    
#%% 

#plotAllFoldsROC(df_folds)

#fig.clear(all)
# =============================================================================
# #%% plotting each contour with a timer
# contourPlot(xx,yy,log_rr,fold_count,'',data=None)
# plotDB(contSet[1],n_X_test,f_X_test,l='+',pause_time=2)
# #%%
# contourPlot(xx,yy,log_rr,fold_count,'',data=None)
# 
# #%%
# contourPlot(xx,yy,zz_f,fold_count,data=f_X_train,'f_X_train')
# contourPlot(xx,yy,zz_n,fold_count,data=n_X_train,'n_X_train')
# #%%
# surfacePlot(xx,yy,log_rr,fold_count)
# =============================================================================
#%% Data preparation

# Data after running Fisher's Linear Discriminant for hand labels
normal  = np.real(normal)
faulty = np.real(faulty)


# Data after running PCA on hand labelled P curve
#normal = norm.values
#faulty = fnorm.values


# prepare data for cross validation
fld_data = np.vstack((normal,faulty))
X = fld_data[:,[0,1]]
y = fld_data[:,2]




#%% Initialise parameters to suit data 

# Change mesh so it comrresponds to dimensions of data space
xmesh = [-0.51,0.2,1000j]                                                        # MAKE SURE CORRECT BY LOOKING AT MIN AND MAX OF SUBSPACE AXES
ymesh = [-0.51,0.07,1000j]


# Instantiate stratified kfold cross validation
k = 5                                                                          # increase this to 10 fold to see any changes
cv = StratifiedKFold(n_splits=k) 

# dataframe to store results for each fold
df_folds = []


fold_count = 0
for train, test in cv.split(X, y):

    # Create current train and test data
    X_train = X[train,:]
    X_test = X[test,:]
    y_train = y[train]
    y_test = y[test]
    
    # faulty and normal data for training and testing for the current fold
    f_X_train = X_train[np.where(y_train==0),:][0]
    n_X_train = X_train[np.where(y_train==1),:][0]
    
    f_X_test = X_test[np.where(y_test==0),:][0]
    n_X_test = X_test[np.where(y_test==1),:][0]
    
    
    
    # Log-likelihood on training data --------------------------------------
    
    # bandwdith for faulty data
    fb = findKDEbandwidth(np.linspace(0, 0.09,200),f_X_train,'Faulty')
    
    # Perform kDE on faulty using using dimensions of plot
    xx, yy, zz_f = kde2D(f_X_train[:,0],
                         f_X_train[:,1],
                         bandwidth=fb.best_params_['bandwidth'],
                         x_mesh=xmesh,                               
                         y_mesh=ymesh)
    
    # Anything below 10^(-5) is 10^(-5)
    zz_f = levelKDE(zz_f)
    
    # bandwdith for normal data
    fb = findKDEbandwidth(np.linspace(0.0, 0.055,50),n_X_train,'Normal')           # need to tune bandwidth search range for difference data
                                                                                # for cv= 5, let np.linspace(0, 0.02,20) to speed up
    # Perform kDE on normal using using dimensions of plot
    xx, yy, zz_n = kde2D(n_X_train[:,0],
                         n_X_train[:,1],
                         bandwidth=fb.best_params_['bandwidth'],
                         x_mesh=xmesh,                               
                         y_mesh=ymesh)
    
    # Anything below 10^(-5) is 10^(-5)
    zz_n = levelKDE(zz_n)
    
    
    # Log likelihood of the relative risk
    log_rr = np.log(zz_f/zz_n)
    
    
    # plot
    #contourPlot(xx,yy,log_rr,fold_count,'',data=None)                                       # make it so that all plots are shown in a single figure
    #surfacePlot(xx,yy,log_rr,fold_count)
    
    
    #df_folds.append(evalutateThresholds(xx,yy,log_rr,
    #                                    normal = n_X_test,
    #                                    faulty = f_X_test,
    #                                    k = fold_count+1))
    
    
    # Evaluation--------------------------------------------------------------
    
    # All levels to iterate through: from -ve to +ve
    #all_lvls = np.array([np.linspace(0.01,np.max(log_rr),100) ]) # positive levels
    #all_lvls = np.array([np.linspace(np.min(log_rr),-0.001,100)]) # negative levels
    all_lvls = np.array([np.linspace(np.min(log_rr),-0.001,100),np.linspace(0.001,np.max(log_rr),100) ])


    # create df
    cols = ['Level','Fold','TN','FN','TP','FP','FPR','TPR','Total','# of datapoints per test']
    df_conM = pd.DataFrame(columns=cols)
    
    # iterate through -ve levels, then +ve levels
    for lvls in all_lvls: 
        
        
        # creating contour object to access contour coords
        CS = ax.contour(xx,yy,log_rr,levels=lvls)
        
        
        # close plot the is created
        plt.close('all')
        
        # Contour Set: all contours from each level
        contSet= CS.allsegs
        #break # for dedugging
    
    #if(fold_count==0):
    #    break # for dedugging
    #fold_count+=1 # delete - just for debugging
        #%%
        # and each group of contour in that level
        for lvl in range(len(contSet)):

            TP = 0  # Reset true positives for current level [normal data]
            FP = 0  # Reset false positives for current level [normal data]
            TN = 0  # Reset true negatives for current level [faulty data]
            FN = 0  # Reset false positives for current level [normal data]
            
            
            # Are any of the contours inside each other? If so which ones
            cont_num = range(len(contSet[lvl])) # number of contours            -----------
            conts_inside = []       # list for contours within each other       This can be a function that takes contSet[lvl] as an input
            for c in itertools.combinations(cont_num,2):
                # check if contours are inside each other
                contour1 = contSet[lvl][c[0]]
                contour2 = contSet[lvl][c[1]]
                #print(c)
                path1 = mpltPath.Path(contour1)
                path2 = mpltPath.Path(contour2)
                
                if(path1.contains_path(path2)):
                    # path2 is inside path1
                    conts_inside.append(c)
                    
                elif(path2.contains_path(path1)):
                    # path1 is inside path2                                     and returns conts_inside
                    conts_inside.append(c) #                                    ------------                                     
            #break
            #%%
            
            
            for contour in range(len(contSet[lvl])):                            # goes through every contour at current level
                
                # reset flag so that 
                flag = 0
                
                #list for encapsulating contour
                encap_cont = []
                
                # Create a path for the current contour
                path = mpltPath.Path(contSet[lvl][contour])
                
                # check if at one contour inside another one
                for i in conts_inside:                                          # likely to only be 1 contour inside another but a for loop is used just incase there is more than 1
                    
                    if(i[0]==contour): # check if the current contour is the ones that encapulates other contours
                        
                        # set flag so that points not counted again
                        flag = 1
                        
                        # all points within add them [already got that code]     ----------------------
                        f_inside = []                                            # function that takes a TN,FN...,path and returns TN,FN...   
                        n_inside = []
                      
                        if((lvls[lvl] < 0) and not(i[0] in encap_cont)):                                                  # -ve levels
                            # Check to see which normal points are within contour
                            n_inside = path.contains_points(n_X_test)
                            
                            # Check to see which faulty points are within contour
                            f_inside = path.contains_points(f_X_test)
                            
                            # number of normal points in contour
                            TP += np.sum(n_inside)
                            
                            # number of faulty points in contour
                            FP += np.sum(f_inside)
                            #print('here1')
                            
                            
                            
                        elif((lvls[lvl] > 0) and not(i[0] in encap_cont)):                                                           # +ve levels
                            
                            # Check to see which faulty points are within contour
                            f_inside = path.contains_points(f_X_test)
                            
                            # Check to see which normal points are within contour
                            n_inside = path.contains_points(n_X_test)
                            
                            # number of faulty points in contour
                            TN += np.sum(f_inside)
                            
                            # number of normal points in contour
                            FN += np.sum(n_inside)                              # ---------------------- function returns TN,FN...
                            #print('here2')
                        
                        # add contour number to list so that it's not added again
                        encap_cont.append(i[0])
                        
                        # next iteration
                        continue
                        
                    if(i[1]==contour): # check if inner contour is current contour
                                                                                    # -----------------------
                        # set flag so that points not counted again
                        flag = 1
                        
                        # values inside contour need to be subtracted           same function but subtracts
                        if(lvls[lvl] < 0):                                                  # -ve levels
                            # Check to see which normal points are within contour
                            n_inside = path.contains_points(n_X_test)
                            
                            # Check to see which faulty points are within contour
                            f_inside = path.contains_points(f_X_test)
                            
                            # number of normal points in contour
                            TP -= np.sum(n_inside)
                            
                            # number of faulty points in contour
                            FP -= np.sum(f_inside)
                            #print('here1')
                            
                            
                            
                        elif(lvls[lvl] > 0):                                                           # +ve levels
                            
                            # Check to see which faulty points are within contour
                            f_inside = path.contains_points(f_X_test)
                            
                            # Check to see which normal points are within contour
                            n_inside = path.contains_points(n_X_test)
                            
                            # number of faulty points in contour
                            TN -= np.sum(f_inside)
                            
                            # number of normal points in contour
                            FN -= np.sum(n_inside)                              # function returns TN,FN...
                            #print('here2')                                     ------------------------
                    
                    
                # counts points inside contour - same function as before as before
                # all points within add them [already got that code]     ----------------------
                f_inside = []                                                   # function that takes a TN,FN...,path and returns TN,FN...   
                n_inside = []
                
                # if no inner contours - if flag not set - go into [i.e. flag is set if contour has been previously counted]
                if(flag==0):
                    if(lvls[lvl] < 0):                                                  # -ve levels
                        # Check to see which normal points are within contour
                        n_inside = path.contains_points(n_X_test)
                        
                        # Check to see which faulty points are within contour
                        f_inside = path.contains_points(f_X_test)
                        
                        # number of normal points in contour
                        TP += np.sum(n_inside)
                        
                        # number of faulty points in contour
                        FP += np.sum(f_inside)
                        #print('here1')
                        
                        
                        
                    elif(lvls[lvl] > 0):                                                           # +ve levels
                        
                        # Check to see which faulty points are within contour      
                        f_inside = path.contains_points(f_X_test)                  
                        
                        # Check to see which normal points are within contour
                        n_inside = path.contains_points(n_X_test)
                        
                        # number of faulty points in contour
                        TN += np.sum(f_inside)
                        
                        # number of normal points in contour
                        FN += np.sum(n_inside)                              # ---------------------- function returns TN,FN...
                        #print('here2')
                        
                        
            # calculate points outside contours
            if(lvls[lvl] < 0):                                                      # -ve levels
                FN = n_X_test.shape[0] - TP                                       # number of normal points outside contour
                TN = f_X_test.shape[0] - FP                                       # number of faulty points outside contour
            elif(lvls[lvl] > 0):                                                                  # +ve levels
                FP = f_X_test.shape[0] - TN                                       # number of faulty points outside contour
                TP = n_X_test.shape[0] - FN                                       # number of normal points outside contour
            
            # Add to dataframe
            df_conM.loc[lvls[lvl],'Level'] = lvl
            df_conM.loc[lvls[lvl],'Fold'] = fold_count
            df_conM.loc[lvls[lvl],'TN'] = TN
            df_conM.loc[lvls[lvl],'FN'] = FN
            df_conM.loc[lvls[lvl],'FP'] = FP
            df_conM.loc[lvls[lvl],'TP'] = TP
            df_conM.loc[lvls[lvl],'Total'] = TP+FP+TN+FN
            df_conM.loc[lvls[lvl],'# of datapoints per test'] = n_X_test.shape[0]+f_X_test.shape[0]

            # If ZeroDivisionError add NaN
            if(((FP==0) and (TN==0))):
                df_conM.loc[lvls[lvl],'FPR'] = np.nan
            else:
                df_conM.loc[lvls[lvl],'FPR'] = FP/np.sum([FP,TN])
                
            # If ZeroDivisionError add NaN
            if(((TP==0) and (FN==0))):
                df_conM.loc[lvls[lvl],'TPR'] = np.nan
            else:
                df_conM.loc[lvls[lvl],'TPR'] = TP/np.sum([TP,FN])
    
                #break # stop after first contour
            #break # stop after first level
        #break # stop after +/- level
    df_folds.append(df_conM)
    if(fold_count==0):break # stop after specified fold
    
    # increment fold count
    fold_count+=1

# concatenate dataframes
df_rr_results = pd.concat(df_folds)


# Contour plot for ending fold that shows the train data too - alter by changing end break statement
#contourPlot(xx,yy,log_rr,fold_count,'',data=None)
#plt.scatter(f_X_train[:,0],f_X_train[:,1],marker='x',s=50,color='gold',linewidth=0.9) # faulty points
#plt.scatter(n_X_train[:,0],n_X_train[:,1],marker='^',s=5,facecolor='None',edgecolors='darkblue',linewidths=0.2) # faulty points


#%% Plot ROC curves for each fold
#plotAllFoldsROC(df_folds)


#%% Average ROC form all folds
#plotAverageROC(df_folds)


#%% Plotting decision boundary at specified level_idx of current contSet variable
#plotCleanDB(contSet,1,f_X_test,n_X_test,'+')


#contourPlot(xx,yy,log_rr,fold_count,'',data=None)                                       # make it so that all plots are shown in a single figure

#surfacePlot(xx,yy,log_rr,fold_count)

#animateSurface(xx,yy,log_rr,fold_count)

    

#%%

#contourPlot(xx,yy,zz_f/zz_n,fold_count,'',data1=f_X_train,data2=n_X_train)                                       # make it so that all plots are shown in a single figure
#contourPlot(xx,yy,log_rr,fold_count,'',data1=f_X_train,data2=n_X_train)     
#%%
#fig,ax = plt.subplots(1,2,squeeze=False,figsize=(12,5))
#cf = ax[0][0].contourf(xx,yy,zz_f/zz_n,levels=15,cmap=cm.plasma)   
#ax[0][0].scatter(n_X_train[:,0], n_X_train[:,1], marker = 'o', s=20,edgecolor='black',linewidth=0.7,facecolor='None')
#ax[0][0].scatter(f_X_train[:,0], f_X_train[:,1], marker = 'x', s=70,c='white',linewidth=1.5)
#fig.colorbar(cf)
#ax.set_title("Faulty/Normal Log-likelihood: Fold {0}".format(fold_count+1))
#ax[0][0].tick_params(axis='both',labelsize=12)
#ax[0][1].tick_params(axis='both',labelsize=12)

#%%
#fig.savefig('/Users/thomasdrayton/Desktop/LR_vs_logLR.png',format='png',dpi=400)



#%%

# =============================================================================
# 
# # Create some random data, I took this piece from here:
# # http://matplotlib.org/mpl_examples/mplot3d/scatter3d_demo.py
# def randrange(n, vmin, vmax):
#     return (vmax - vmin) * np.random.rand(n) + vmin
# n = 100
# xx = randrange(n, 23, 32)
# yy = randrange(n, 0, 100)
# zz = randrange(n, -50, -25)
# 
# # Create a figure and a 3D Axes
# fig = plt.figure()
# ax = Axes3D(fig)
# 
# # Create an init function and the animate functions.
# # Both are explained in the tutorial. Since we are changing
# # the the elevation and azimuth and no objects are really
# # changed on the plot we don't have to return anything from
# # the init and animate function. (return value is explained
# # in the tutorial.
# def init():
#     ax.scatter(xx, yy, zz, marker='o', s=20, c="goldenrod", alpha=0.6)
#     return fig,
# 
# def animate(i):
#     ax.view_init(elev=10., azim=i)
#     return fig,
# 
# # Animate
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=360, interval=20, blit=True)
# 
# anim.save('/Users/thomasdrayton/Desktop//basic_animation.mp4', fps=30, writer='ffmpeg')
# =============================================================================


#%%


def init():
    ax.plot_surface(xx,yy,log_rr,antialiased=False,
                 cmap=cm.plasma,
                 alpha=1)
    return fig,


def animate(i):
    ax.view_init(elev=30., azim=i)
    return fig,


    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
#%
anim.save('/Users/thomasdrayton/Desktop//basic_animation_600.mp4', fps=30, dpi=600,bitrate=10000)
 
