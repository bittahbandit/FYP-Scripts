#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:11:29 2018


contains plotSubset function data that plots the data from  the 
"Data Separated into Varibles" folder but only plots a specified number of
test plots - the function used to create the subplot figures in "SubplotForEachVar"


@author: thomasdrayton
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

#---------------------------------------------------------------------------------------

'''
plotSubset function data that plots the data from  the filepath, f, by plotting
each column onto a single plot
    - the plot is named as the csv file that it accesses with the .csv removed
    - the columns are plotted chronologically
    - you choose the number of columns to plot using the parameter r (default=10)
    - the file is saved as a png in a location of choice by modifying the 1st
      parameter in fig.savefig

'''


def plotSubset(f,r=10) :
    ''' f is the pathname of the data'''
    #print(type(file))
    df = pd.read_csv(f)
    
    #plot
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time(s)',fontsize = 6)
    ax.tick_params(axis = 'both',labelsize = 5)
    ax.set_title(f.replace(".csv",""),fontsize = 6)    
    for col in df.columns[0:r]:     # iterating through each of the columns 
        ax.plot(df.index,col,data=df, linewidth=0.11)
        
    png = f.replace(".csv",".png") # creates the correct expension for png in next line
    fig.savefig("/Users/thomasdrayton/Desktop/FYP/Code/Figures/SubsetForEachVar/"+png,bbox_inches='tight',dpi = 1800) 
    #fig.clf()   # this should be commented out if you want to create different plots using this same function



def plot_var_from_test(var,test,n1v = False):
    df = pd.read_csv(test)
    
    #plot
    fig,ax = plt.subplots(1,1,figsize=(16,10))
    ax.set_xlabel('Time(s)',fontsize = 14)
    ax.set_ylabel('Signal Magnitude',fontsize = 14)
    ax.tick_params(axis = 'both',labelsize = 11)
    #ax.set_title(test.replace(".csv",""),fontsize = 10)
    #ax.plot(df.index,df.loc[:,var],linewidth=0.5,label='{0}'.format(var))
    if(n1v==True):
        ax.plot(df.index,df.loc[:,'n1v'],linewidth=0.9,label='n1v')
    ax.plot(df.index,df.loc[:,'n2v'],linewidth=0.59,label='n2v')
    ax.plot(df.index,df.loc[:,'n3v'],linewidth=0.9,label='n3v')
    ax.legend(fontsize=15,title='Shaft Speed')
    ax.get_legend().get_title().set_fontsize('14')



    

# =============================================================================
# =============================================================================


'''
program to either:
    
    - create plots for all the csv files in Data Separated into Varibles
    - create singel csv file

plotSubset function is used
'''


#indir ="/Users/thomasdrayton/Desktop/FYP/Code/Data Separated into Varibles" #make sure path is correct
#os.chdir(indir) # enter VarSeparatedData folder
#fileList = glob.glob("*.csv")   # putting all the csv files into list

# =============================================================================
# # plotting single
# plotSubset('trav.csv',1)
# =============================================================================



# # Plotting all 
# 
# for file in fileList:
#     #print(file.replace(".csv",".png"))
#     #print(file.index())
#     plotSubset(file)



# =============================================================================
# =============================================================================    

indir ="/Users/thomasdrayton/Desktop/FYP/Code/All Normalised Data" #make sure path is correct
os.chdir(indir) # enter normalised data folder
fileList = glob.glob("*.csv")


plot_var_from_test('tcarv_S','T123411.csv',n1v = True)
