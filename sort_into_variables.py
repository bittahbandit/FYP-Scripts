#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:15:51 2018

-   Creates a csv file with all information from that extracts data from a specified column of each file in
    "Normalised Data" folder, and creates
    
-   Also program to plot specified varibles from Normalised Data directory 

@author: thomasdrayton
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# Change varible to the one you want
var = "tfuelv"
                                                                                                                                
def concatenate(indir ="/Users/thomasdrayton/Desktop/FYP/Code/Normalised Data", outfile = "/Users/thomasdrayton/Desktop/FYP/Code/VarSeparated/"+var+".csv") :
    os.chdir(indir)                 # setting wkdir
    fileList = glob.glob("*.csv")   # putting all the csv files into list
    dfList = []                     # initalising empty list for column to be added into
    for filename in fileList:
        #print(filename)
        df = pd.read_csv(filename)  # create dataframe from csv file
        dfList.append(df[var])    # add column into list           NEEDS TO BE CHANGED TO MATCH VARIBLE
                
    #printf(dfList)
    df = pd.DataFrame(dfList)       # create dataframe from list
    df = df.transpose()             # transpose to set headers at top
    df.columns = fileList           # assign correct column names to headers
    print(df.head(20))
    print(df.shape)
    
    df.to_csv(outfile,index = None,header = fileList) # create csv file in the outfile pathname




# =============================================================================
# concatenate()           # calling the function
# =============================================================================


# 
# Comparison of selected varibles

indir ="/Users/thomasdrayton/Desktop/FYP/Code/Normalised Data" #make sure path is correct
os.chdir(indir) # enter normalised data folder
fileList = glob.glob("*.csv")

count = 0

for f in fileList:
    df = pd.read_csv(f)
    selection = ['n1v','IP_Vib_Magnitude_A']#['n1v','tfuelv','toilv','HP_Vib_Magnitude_A','tgtv','p30v']
    ss = df[selection]
    #print(ss.head())
    #print(ss.columns.tolist())
    
    # plot
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time(s)',fontsize = 6)
    ax.tick_params(axis = 'both',labelsize = 5)
    ax.set_title(f.replace(".csv",""),fontsize = 6)    
    for col in ss.columns:     # iterating through each of the columns 
        ax.plot(ss.index,col,data=ss, linewidth=0.11)
    ax.legend(ss.columns.tolist(), loc = 'upper right',fontsize = 4)
    
    png = f.replace(".csv",".png") # creates the correct expension for png in next line
    fig.savefig("/Users/thomasdrayton/Desktop/FYP/Code/Figures/Speed IP Vibration comparison/"+png,bbox_inches='tight',dpi = 1400) 
    fig.clf()
    count+=1
    print(count)
    

