# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 01:01:43 2023

@author: SATYA PRAKASH
"""

import os
import pandas as pd
import numpy as np



os.chdir("D:/spyder")
data_csv = pd.read_csv('flavors_of_cocoa.csv', index_col=0)
print(data_csv)

"""
In this File information stored are ---
1. Import data -- already done
2. Concise summary of dataframe
3. Converting variable's data types
4. Category vs Object data types 
5. cleaning column 'Doors
6. Getting count of missing values
7.  

"""

print("changing the data type")
"""
 astype() is used to explicitly convert data types from one to another
 
"""
data_csv['Rating'] = data_csv['Rating'].astype('object')
print(data_csv)

"""
 nbytes() is used to get the total bytes consumed by the elements of the column

"""
print(data_csv['Company'].nbytes)
print(data_csv.info())

"""
 To replace the unique value present in the particular column
 syntax is =====  Dataframe.replace([to_replace, value, inplace=True])
"""
data_csv['Company'].replace('Zotter', "Satya Prakash", inplace=True )
print(data_csv)


"""
 DataFrame.isnull.sum() is used to check the count of missing value present in the column
 

"""
