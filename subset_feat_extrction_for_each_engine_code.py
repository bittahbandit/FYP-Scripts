#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:17:36 2018

    - 1st file to run to create database
    
    - make sure outfile is clear [/Users/thomasdrayton/Desktop/FYP/InformationExtraction/]
    
    - Then to concatenate files to create database, run db_creatation_from_feat_extrction_file

@author: thomasdrayton
"""

import os
import glob
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import more_itertools as mit




# =============================================================================
# =============================================================================
# Function definitions
    
def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


# =============================================================================
# =============================================================================

indir ="/Users/thomasdrayton/Desktop/FYP/Code/All Normalised Data"
os.chdir(indir)

fileList = glob.glob("*.csv")   # putting all the csv files into list

fileNum = 0

exld = 0

for file in fileList:
    fileNum+=1
    df = pd.read_csv(file)
    print(file,fileNum)
    selection = ['eng test','window','n1v_f','tfuelv_f','toilv_f',
                 'LP_Vib_Magnitude_A_f','IP_Vib_Magnitude_A_f',
                 'HP_Vib_Magnitude_A_f','t20v_f','t25v_f','t30v_f','p20v_f',
                 'ps26v_f','p30v_f','P42_f','P44_f','tgtv_f','tcarv_S_f',
                 'tcafv_S_f','tprtrimmed_f','trav_f','poilv_S_f','tsasv_S_f']
    db = pd.DataFrame(columns=selection, dtype = float)
    
    df['dn1v'] = np.gradient(df['n1v'].rolling(center=True,window=50).mean())
    df['dn1v'] = df['dn1v']*10
    
    df.drop(df[(df['dn1v'] < -0.001) | (df['dn1v'] > 0.001)].index, inplace=True)
    
    l_withInts = list(find_ranges(df.index))
    
    l = []
    for i in l_withInts:
        if(isinstance(i,tuple)):
            l.append(i)
            
            
    cutoff = 80
    cut = []
    for i in l:
        diff = i[1] - i[0]
        if(diff<cutoff):
            cut.append(i)
        
    l = [x for x in l if x not in cut]
    
    
    # Find the highest speed in the data from the tuples from the actual data
    window_speed = []
    for i in l:
        initial_val = i[0]
        window_speed.append(df.loc[initial_val,'n1v'])  
        
        highest_speed = max(window_speed)                  
        
    idx = 0
    for i in l:
        if (df.loc[i[0],'n1v'] == highest_speed):
            break
        idx+=1

    # Only keep 7 steps including the top one
    l = l[idx:idx + 7]
    
    # Excluding profiles where the starting value is less 0.9 for n1v
    if(df.loc[l[0][0],'n1v']<0.9):
        exld+=1
        continue
    
# =============================================================================
#     #add windows as columns to initialise db
#     for i in range(len(l)):
#         db[i] = np.nan
# =============================================================================
        
    count = 0
    # Iterating through each window w
    for w in range(len(l)):
        
        win = l[w] #window tuple 
        win_range = range(win[0],win[1]) # creating range from window tuple
        
        mid = int(win[1] - (win[1]-win[0])/2)  # find middle index value

    
        #n_featrue
        n1v_f = df.loc[win_range,'n1v'].mean()
        db.loc[w,'n1v_f'] = n1v_f
    
        # tfuelv
        tfuelv_f = df.loc[win_range,'tfuelv'].mean()   
        db.loc[w,'tfuelv_f'] = tfuelv_f
    
        # toilv                         
        toilv_f = df.loc[win_range,'toilv'].mean()           
        db.loc[w,'toilv_f'] = toilv_f
                
        # IP_Vib_Magnitude_A
        IP_Vib_Magnitude_A_f = df.loc[win_range,'IP_Vib_Magnitude_A'].mean()            
        db.loc[w,'IP_Vib_Magnitude_A_f'] = IP_Vib_Magnitude_A_f
        
        # LP_Vib_Magnitude_A
        LP_Vib_Magnitude_A_f = df.loc[win_range,'LP_Vib_Magnitude_A'].mean()            
        db.loc[w,'LP_Vib_Magnitude_A_f'] = LP_Vib_Magnitude_A_f
        
        # HP_Vib_Magnitude_A
        HP_Vib_Magnitude_A_f = df.loc[win_range,'HP_Vib_Magnitude_A'].mean()            
        db.loc[w,'HP_Vib_Magnitude_A_f'] = HP_Vib_Magnitude_A_f
        
        # t20v
        t20v_f = df.loc[win_range,'t20v'].mean()     
        db.loc[w,'t20v_f'] = t20v_f
        
        # t25v
        t25v_f = df.loc[win_range,'t25v'].mean()     
        db.loc[w,'t25v_f'] = t25v_f
        
        # t30v
        t30v_f = df.loc[win_range,'t30v'].mean()     
        db.loc[w,'t30v_f'] = t30v_f
        
        # p20v
        p20v_f = df.loc[win_range,'p20v'].mean()  
        db.loc[w,'p20v_f'] = p20v_f
        
        # ps26v
        ps26v_f = df.loc[win_range,'ps26v'].mean()  
        db.loc[w,'ps26v_f'] = ps26v_f
        
        # p30v
        p30v_f = df.loc[win_range,'p30v'].mean()  
        db.loc[w,'p30v_f'] = p30v_f
        
        # P42
        P42_f = df.loc[win_range,'P42'].mean()  
        db.loc[w,'P42_f'] = P42_f
        
        # P44
        P44_f = df.loc[win_range,'P44'].mean()  
        db.loc[w,'P44_f'] = P44_f
        
        # tgtv
        tgtv_f = df.loc[win_range,'tgtv'].mean()           
        db.loc[w,'tgtv_f'] = tgtv_f
        
        # tcarv_S
        tcarv_S_f = df.loc[win_range,'tcarv_S'].mean()  
        db.loc[w,'tcarv_S_f'] = tcarv_S_f
        
        # tcafv_S
        tcafv_S_f = df.loc[win_range,'tcafv_S'].mean()  
        db.loc[w,'tcafv_S_f'] = tcafv_S_f
        
        # tprtrimmed
        tprtrimmed_f = df.loc[win_range,'tprtrimmed'].mean()  
        db.loc[w,'tprtrimmed_f'] = tprtrimmed_f
        
        # trav
        trav_f = df.loc[win_range,'trav'].mean()  
        db.loc[w,'trav_f'] = trav_f
        
        # poilv_S
        poilv_S_f = df.loc[win_range,'poilv_S'].mean()  
        db.loc[w,'poilv_S_f'] = poilv_S_f
        
        # tsasv_S
        tsasv_S_f = df.loc[win_range,'tsasv_S'].mean()  
        db.loc[w,'tsasv_S_f'] = tsasv_S_f
        
        # adding engine test code
        db.loc[w,'eng test'] = file.replace(".csv","")
    
        # adding window
        db.loc[w,'window'] = count
        count+=1
        
    


    
    outfile = "/Users/thomasdrayton/Desktop/FYP/InformationExtraction/"+file
    db.to_csv(outfile)

print('Excluded: ',exld/fileNum*100,'%')

