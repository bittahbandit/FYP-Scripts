#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:48:20 2019

Convert youtube transcript into a paragraph

@author: thomasdrayton
"""


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

script = open("utub2para.txt",'r')

out = ''

for line in script:
    if(is_number(line[0])):
        continue
    out+=line
    
print(out)