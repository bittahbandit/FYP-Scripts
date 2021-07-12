#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:15:51 2018

Creates a csv file with all information from that extracts data from a specified column of each file in
"Normalised Data" folder, and creates

@author: thomasdrayton
"""

import os
import glob
import pandas as pd

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


concatenate()           # calling the function

