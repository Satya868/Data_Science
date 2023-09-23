# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 23:32:21 2023

@author: Satya Prakash
"""


""" 
Here we will use the Looping Condition to categorise the Data as old or new

"""

import os
import pandas as pd
import numpy as np

os.chdir("D:/spyder")
data_csv = pd.read_csv('flavors_of_cocoa.csv', index_col=0)
print(data_csv)


print("BELOW IS THE CODE TO CATEGORISE IT IN OTHER CLASS")
for i in range(1, len(data_csv['Rating']), 1):
    if(data_csv['Rating'][i]<=3.0):
        data_csv['Review Rev'][i]="Bad"
    elif(data_csv['Rating'][i]>=3.3):
        data_csv['Review Rev'][i]="Good"
    else:
        data_csv['Review Rev'][i]="Medium"
        
print(data_csv)  

"""
  Below line of code counts the total counts of review under particular sub name
"""    
print("Counting the Total number under particular sub-head")
print(data_csv['Rating'].value_counts())  


"""
 Another control structure in the python can be use of def funct()   etc
  here we can cinvert months into years  by // with 12, here we can round off the value 
  to one decimal place
  
  
  though we cant do this with .csv file available with me
  
"""

