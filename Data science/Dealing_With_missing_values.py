# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 02:19:03 2023

@author: SATYA PRAKASH
"""

import os
import pandas as pd
import numpy as np


os.chdir("D:/spyder")
data_csv = pd.read_csv('flavors_of_cocoa.csv', index_col=0, na_values=["??", "????"])
print(data_csv)

# creating copies
fl_data = data_csv.copy()
f_d = fl_data.copy()

"""
         IDENTIFYING MISSIG VALUES
         1. isnull() and isna() are used == in pandas datframe
         2. these functions returns a datsframe of boolean values which are True for NaN
           values
           
"""
# To check the count of missing values present in each column
print(f_d.isna().sum()) # 2nd way is written down
print(f_d.isnull().sum())

# subsetting the rows that have one or more missing values
missing = f_d[f_d.isnull().any(axis=0)]
print(missing)


"""
  APPROACH TO FILL THE MISSING VALUES
  2 Approaches are
    1. Fill the missing values by MEAN/MEDIAN in case of numerical values
    2. fill the missing values with Maximum count in case of categorical variable
    
"""
"""
 We can decide it has to be filled with mean or median by looking at the its
 description using SYNTAX -- dataFrame.describe()
 
 it excludes NaN values, we should use MEDIAN generally
"""

print(f_d.describe())


# CAlculating mean of rating Variable
print("CAlculating mean of rating Variable")
print(f_d['Rating'].mean())


# Filling the missing values in "Rating" column
print("Filling the missing values in Rating column")
f_d['Rating'].fillna(f_d['Rating'].mean(), inplace=True)


# Calculating medain values
print(" mEdiaN vAlueS of Data")
print(f_d['Rating'].median())

# Filling the missing data with NA/NaN values

f_d['Rating'].fillna(f_d['Rating'].median, inplace=True)
f_d['Review Date'].fillna(f_d['Review Date'].median, inplace=True)
print(f_d[f_d.isnull().any(axis=1)])

"""
TO FILL THE NA/NaN values in both numerical and categorical variable at one stretch
"""
f_d = f_d.apply(lambda x:x fillna(x.mean()) if x.dtypes == 'float' else x.fillna(x.value_counts().index[0]))



print("checking the missing DATA")
print(f_d.isnull().sum())
