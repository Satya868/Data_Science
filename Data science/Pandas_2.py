# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 23:50:45 2023

@author: Satya Prakash
"""

import os
import pandas as pd
import numpy as np



os.chdir("D:/spyder")
data_csv = pd.read_csv('flavors_of_cocoa.csv', index_col=0)
print(data_csv)

"""
  here we will learn about
  1. Data types 
     a. Numeric -- include integer and float
     b. Character -- Strings are known as object in pandas
  2. Checking Data types of each column 
  3. Count of unique data types
  4. Selecting Data based on data types
  5. Concise summary of dataframe   
  6. Checking format of each column
  7. Getting unique elements of each column
  8. 
  
"""  
print(data_csv.dtypes)
# get_dtype_counts() returns counts of unique data types in the dataframe

#print(data_csv.dtypes()) ---> in latest version of pandas some shitty people 
# played a shit game with its code on the name of open source

print("Printing only specific types of column")
print(data_csv.select_dtypes(exclude=[float]))


"""
info() returns concise summary of a dataframe 
    * data types of index
    * data types of column
    * count of non-null values
    * memory usages

"""   
print("______________________________________________")
print("Printing the info of particular data frame")
print(data_csv.info())





"""
unique() is used to find the unique elements of a column
syntax --- np.unique(array)

"""
print("----------------------------------------------------")
print("Printing the unique Name in any particular column")
print("----------------------------------------------------")
print(np.unique(data_csv['Company']))


