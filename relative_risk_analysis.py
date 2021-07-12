#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:46:08 2018

    - Kernel Density Estimatation
    
        - Uses Cross validation to select the bandwidth

    - Relative Risk ratio of Faulty over Normal
    
    - Takes the log of the ratio
    
    - Creates an ROC curve going through different levels/contours/thresholds
        - Deos it fall all the data which is not correct
        - Need to do use cross validation 
        
        - see _______ for correct evaluation
    
    - NEED TO RUN FISHER'S LINEAR DISCRIMINANT to get data from subspace
    
@author: thomasdrayton
"""

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


# Change to latex font ------------------------------------------------
plt.rcParams["font.family"] = "serif"

#%%


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


#%%



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
    return xx, yy, np.reshape(z, xx.shape)



# ----------------------------------------------------------------------------


# consistent plot zoom
x_min = 0.05
x_max = 0.55
y_min = 0.3
y_max = -0.7



# ----------------------------------------------------------------------------



# Faulty 

# bandwidth hyperparameter gridsearch using 
bandwidths = np.linspace(0, 0.5,200)
gridS = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=KFold(5),
                    iid=False,
                    n_jobs=-1)
gridS.fit(faulty[:,0:2])
print('Faulty: ',gridS.best_params_)                                                       # Print best bandwidth



# Perform kDE on faulty using using dimensions of FLD plot
xx, yy, zz_f = kde2D(faulty[:,0],
                   faulty[:,1],
                   bandwidth=gridS.best_params_['bandwidth'],
                   x_mesh=[-0.2,0.2,1000j],                                     # need to change depending on the axes of data 
                   y_mesh=[-0.29,00.1,1000j])                                         # need to change depending on the axes of data 

# anything below 10^(-5) is 10^(-5)
for i,vals in enumerate(zz_f):
    for ii,val in enumerate(vals):
        if(val <= 10**(-5)):
            zz_f[i][ii] = 10**(-5)

#%% Faulty plots

fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})

# surface plot
ax.plot_surface(xx,yy,zz_f, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.plasma)
ax.scatter(faulty[:,0], faulty[:,1], marker = 'x', s=30,c='red',linewidth=0.7)
ax.set_title("")


#contour plot
fig1,ax1 = plt.subplots(1,1)

cf = ax1.contourf(xx,yy,zz_f,levels=15,cmap=cm.plasma)
fig1.colorbar(cf)
#ax1.set_xlim(x_min,x_max)
#ax1.set_ylim(y_min,y_max)
ax1.set_title("KDE on faulty data")


#%% Normal 

# bandwidth hyperparameter gridsearch using 
bandwidths = np.linspace(0, 0.3,100)
gridS = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=KFold(5),
                    iid=False,
                    n_jobs=-1)
gridS.fit(normal[:,0:2])
print(gridS.best_params_)     
#%%
# Perform kDE on normal using using dimensions of FLD plot
xx, yy, zz_n = kde2D(normal[:,0],
                   normal[:,1],
                   #bandwidth=gridS.best_params_['bandwidth'],
                   bandwidth =0.006060606060606061,
                   x_mesh=[-0.2,0.2,1000j],                                     # MAKE SURE CORRECT BY LOOKING AT MIN AND MAX OF SUBSPACE AXES
                   y_mesh=[-0.29,-0.023,1000j])                                         # need to change depending on the axes of data 


# Anything below 10^(-5) is 10^(-5)
for i,vals in enumerate(zz_n):
    for ii,val in enumerate(vals):
        if(val <= 10**(-5)):
            zz_n[i][ii] = 10**(-5)
            
#%% Normal plots
            
            
            
# surface plot
fig2,ax2 = plt.subplots(1,1,subplot_kw={'projection':'3d'})

ax2.plot_surface(xx,yy,zz_n, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.plasma)
#ax.scatter(normal[:,0], normal[:,1], marker = 'x', s=30,c='red',linewidth=0.7)
ax2.set_title("")

# Contour plot
fig3,ax3 = plt.subplots(1,1)
cf3 = ax3.contourf(xx,yy,zz_n,levels=15,cmap=cm.plasma)
fig3.colorbar(cf3)
#ax3.set_xlim(x_min,x_max)
#ax3.set_ylim(y_min,y_max)
ax3.set_title("KDE on normal data")


    

#%% Likelihood ratio and log-likelihood

# Likelihood
rr =zz_f/zz_n

# Log likelihood of the relative risk
log_rr = np.log(rr)

# Contour plot
fig4,ax4 = plt.subplots(1,1)
cf4 = ax4.contourf(xx,yy,rr,levels=2,cmap=cm.plasma)
ax4.scatter(faulty[:,0], faulty[:,1], marker = 'x', s=30,c='red',linewidth=0.7)
fig4.colorbar(cf4)
ax4.set_title("Faulty/Normal likelihoood")

fig41,ax41 = plt.subplots(1,1)
cf41 = ax41.contourf(xx,yy,log_rr,levels=50,cmap=cm.plasma)
ax41.scatter(faulty[:,0], faulty[:,1], marker = 'x', s=30,c='k',linewidth=0.7)
fig41.colorbar(cf41)
ax41.set_title("Faulty/Normal Log-likelihoood")

#ax41.set_xlim(-0.1,0.55)
#ax41.set_ylim(0.26,1.1)

#ax4.set_xlim(-0.1,0.55)
#ax4.set_ylim(0.26,1.1)


#%% log_rr surface plot
fig5,ax5 = plt.subplots(1,1,subplot_kw={'projection':'3d'})
cb = ax5.plot_surface(xx,yy,log_rr,antialiased=False,
                 cmap=cm.plasma,
                 alpha=1) #ratio

#ax5.contour(xx,yy,log_rr,levels=np.linspace(0,np.max(log_rr),100),cmap=cm.plasma)
ax5.scatter(faulty[:,0], faulty[:,1], marker = 'x', s=30,c='k',linewidth=0.7)

#ax5.set_xlim(x_min,x_max)
#ax5.set_ylim(y_min,y_max)
ax5.set_title("Faulty/Normal log-likelihoood")
fig5.colorbar(cb)


#%% iterating upwards from contour at 0

cols = ['TN','FN','TP','FP','FPR','TPR']
df_conM = pd.DataFrame(columns=cols)

# 100 levels between 0 and max value to iterate through
lvls = np.linspace(0,np.max(log_rr),100) 

# creating contour object to access contour coords
CS = ax.contour(xx,yy,log_rr,levels=lvls)

# close plot the is created
plt.close('all')

# Contour Set: all contours from each level
contSet= CS.allsegs

# for each level and each contour in that level
for lvl in range(len(contSet)): 
    
    TN = 0  # Reset true negatives for current level [faulty data]
    FN = 0  # Reset false positives for current level [normal data]
    TP = 0
    FP = 0
   
    
    for contour in range(len(contSet[lvl])):
        
        # Create a path for the current contour
        path = mpltPath.Path(contSet[lvl][contour])
        
        # Check to see which faulty points are within contour
        f_inside = path.contains_points(faulty[:,0:2])
        
        # Check to see which normal points are within contour
        n_inside = path.contains_points(normal[:,0:2])
        
        # number of faulty points in contour
        TN += np.sum(f_inside)
        
        # number of normal points in contour
        FP += np.sum(n_inside)
        
        # number of faulty points outside contour
        FN += np.sum(~f_inside)
        
        # number of normal points outside contour
        TP += np.sum(~n_inside)
        
        # plot to check results
        #plt.plot(contSet[lvl][contour][:,0],contSet[lvl][contour][:,1])
        #plt.scatter(normal[:,0], normal[:,1], marker = 'x', s=30,c=n_inside,linewidth=0.7)
        #if(lvl==1):
        #    break
        
    # compute FN and TP
    #FN = faulty.shape[0] - TN
    #TP = normal.shape[0] - FP
    

    #break
    # Add to dataframe
    df_conM.loc[lvls[lvl],'TN'] = TN
    df_conM.loc[lvls[lvl],'FN'] = FN
    df_conM.loc[lvls[lvl],'FP'] = FP
    df_conM.loc[lvls[lvl],'TP'] = TP
    
    # If ZeroDivisionError add NaN
    if(((FP==0) and (TN==0)) or ((TP==0) and (FP==0))):
        df_conM.loc[lvls[lvl],'FPR'] = np.nan
        df_conM.loc[lvls[lvl],'TPR'] = np.nan
    else:
        df_conM.loc[lvls[lvl],'FPR'] = FP/(FP+TN)
        df_conM.loc[lvls[lvl],'TPR'] = TP/(TP + FN)
    
    
    
    
#%% For negative levels from min to 0

# 100 levels between 0 and max value to iterate through
lvls = np.linspace(np.min(log_rr),-0.01,100)

# creating contour object to access contour coords
CS = ax.contour(xx,yy,log_rr,levels=lvls)

# close plot the is created
plt.close('all')

# Contour Set: all contours from each level
contSet= CS.allsegs

# for each level and each contour in that level
for lvl in range(len(contSet)): 
    
    TP = 0  # Reset true positives for current level [normal data]
    FP = 0  # Reset false positives for current level [normal data]
    TN = 0
    FN = 0
    
    for contour in range(len(contSet[lvl])):                                    # goes through every contour at current level
        
        # Create a path for the current contour
        path = mpltPath.Path(contSet[lvl][contour])
        
        # Check to see which normal points are within contour
        n_inside = path.contains_points(normal[:,0:2])
        
        # number of normal points in contour
        TP += np.sum(n_inside)
        
        # Check to see which faulty points are within contour
        f_inside = path.contains_points(faulty[:,0:2])
        
        # number of faulty points in contour
        FN += np.sum(f_inside)
        
        # number of positive points outside contour
        FP += np.sum(~n_inside)
        
        # number of faulty points outside contour
        TN += np.sum(~f_inside)
        
        
        
    # compute TN and FN
    #FN = normal.shape[0] - TP                                                   # all normal points that are outside predicted normal contour
    #TN = faulty.shape[0] - FP                                                   # all faulty points that are outside predicted normal contour
    
    # Add to dataframe
    df_conM.loc[lvls[lvl],'TN'] = TN
    df_conM.loc[lvls[lvl],'FN'] = FN
    df_conM.loc[lvls[lvl],'FP'] = FP
    df_conM.loc[lvls[lvl],'TP'] = TP
    
    # If ZeroDivisionError add NaN
    if(((FP==0) and (TN==0)) or ((TP==0) and (FP==0))):
        df_conM.loc[lvls[lvl],'FPR'] = np.nan
        df_conM.loc[lvls[lvl],'TPR'] = np.nan
    else:
        df_conM.loc[lvls[lvl],'FPR'] = FP/(FP+TN)
        df_conM.loc[lvls[lvl],'TPR'] = TP/(TP + FN)
  
#%% sort dataframe

# sort by FPR
#df_conM.sort_values(by='',inplace = True)
        
# sort by index / level
df_conM.sort_index(inplace=True)

# Reset index : level is now a column
df_conM.reset_index(inplace=True)
df_conM.columns = ['Level','TN','FN','TP','FP','FPR','TPR']

#%% ROC curve 
        
fig,ax = plt.subplots()

auc = np.trapz(y=df_conM.loc[:,'TPR'],x=df_conM.loc[:,'FPR'],dx=0.01)

ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),linewidth=1,c='lightgrey')    # 0.5 reference line
ax.scatter(df_conM.loc[:,'FPR'],df_conM.loc[:,'TPR'],linewidth = 1,s = 7,
        label='AUC={0:.2f}'.format(auc))         #
ax.set_xlabel('FPR',fontsize = 8)
ax.set_ylabel('TPR',fontsize = 8)
ax.set_title('ROC curve from Faulty/Normal Log-likelihood')
#ax.legend()

# =============================================================================
# #Adding all test file labels to plots
# for i, txt in enumerate(df_conM.index):
#     ax.annotate(txt, (df_conM.loc[i,'FPR'], df_conM.loc[i,'TPR']),fontsize=10)
# =============================================================================

#%% Individual levels



# creating contour object to access contour coords: same levels as df_conM      # change so the right contour levels are being accessed form dataframe
CS = ax.contour(xx,yy,log_rr,levels=df_conM.Level.values)                       

# creating contour object to access contour coords: same levels as df_folds[0]
#CS = ax.contour(xx,yy,log_rr,levels=df_folds[0].Level.values)

# close plot the is created
plt.close('all')
# Contour Set: all contours from each level
contSet= CS.allsegs


level_idx = 13

# For Faulty
# for 1st contour
path = mpltPath.Path(contSet[level_idx][0])
f_inside1 = path.contains_points(faulty[:,0:2]) # Check to see which faulty points are within contour path
# for 2nd contour - may need to change to index first contour if IndexError
path = mpltPath.Path(contSet[level_idx][0])
f_inside2 = path.contains_points(faulty[:,0:2]) # Check to see which faulty points are within contour path

# All faulty data within is red, all outside is blue
f_inside = []
for i in range(faulty.shape[0]):
    if((f_inside1[i]==True) or (f_inside2[i]==True)):
        f_inside.append('red')
    else:
        f_inside.append('blue')

# change to array for indexing ease for later
f_inside = np.array(f_inside)




# For Normal
# for 1st contour
path = mpltPath.Path(contSet[level_idx][0])
n_inside1 = path.contains_points(normal[:,0:2]) # Check to see which faulty points are within contour path
# for 2nd contour - may need to change to index first contour if IndexError
path = mpltPath.Path(contSet[level_idx][0])
n_inside2 = path.contains_points(normal[:,0:2]) # Check to see which faulty points are within contour


# All normal data within is orange, all outside is purple
n_inside = []
for i in range(normal.shape[0]):
    if((n_inside1[i]==True) or (n_inside2[i]==True)):
        n_inside.append('orange')
    else:
        n_inside.append('purple')

# change to array for indexing ease for later
n_inside = np.array(n_inside)

# NEED TO CHECK LABELS: 
# IF LEVEL≥0,           IF LEVEL<0, 
#   redcross=TN             redcross=FN
#   bluecross=FN            bluecross=TN
#   purpleTri=TP            purpleTri=FP
#   orangeTri=FP            orangeTri=TP

plt.plot(contSet[level_idx][0][:,0],contSet[level_idx][0][:,1],c='k',linewidth=1) #plot first contour
plt.plot(contSet[level_idx][1][:,0],contSet[level_idx][1][:,1],c='k',linewidth=1) # plot 2nd contour
plt.scatter(faulty[np.where(f_inside=='red'),0], faulty[np.where(f_inside=='red'),1], marker = 'x', s=50,c='red',linewidth=0.7,label='FN')
plt.scatter(faulty[np.where(f_inside=='blue'),0], faulty[np.where(f_inside=='blue'),1], marker = 'x', s=50,c='blue',linewidth=0.7,label='TN')
plt.scatter(normal[np.where(n_inside=='purple'),0], normal[np.where(n_inside=='purple'),1],marker = '^',facecolor="None", s=20,edgecolors='purple',linewidths=0.5,label='FP')
plt.scatter(normal[np.where(n_inside=='orange'),0], normal[np.where(n_inside=='orange'),1],marker = '^',facecolor="None", s=20,edgecolors='orange',linewidths=0.5,label='TP')
plt.title('Contour at Faulty/Normal at {0:.2f} Log-likelihood'.format(df_conM.Level.values[level_idx]))
plt.legend()

#%% testing a using all contours at once

# create all contours in single path all paths
all_conts = np.vstack(tuple(contSet[level_idx]))  
path = mpltPath.Path(all_conts,closed = True)   # change closed parameter

# check if faulty inside
f_in = path.contains_points(faulty[:,0:2])

# check if normal inside
n_in = path.contains_points(normal[:,0:2])
n_colors = []
for i in (n_in):
    if(i==True):
        n_colors.append('purple')
    else:
        n_colors.append('orange')
        

plt.plot(all_conts[:,0],all_conts[:,1],linewidth=1,c='k')
plt.scatter(faulty[:,0],faulty[:,1],c=f_in,marker='x',s=50,linewidth = 0.7)
plt.scatter(normal[:,0],normal[:,1],facecolor=None,marker='^',s=30,edgecolors=n_colors,linewidths=0.5)


#%%
 
