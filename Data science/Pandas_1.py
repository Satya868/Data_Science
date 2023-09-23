# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 14:59:38 2023

@author: SATYA PRAKASH

"""
import os
import pandas as pd
import numpy as np



os.chdir("D:/spyder")

"""
  Functions to be performed
  1. Introduction to Pandas
  2. Importing data into sypder(Already done)
  3. Creating copy of original data
  4. Attributes of data
  5. INDEXING and SELECTING DATA
"""

""" By passing index_col=0, first column becomes the index column
"""


data_csv = pd.read_csv('flavors_of_cocoa.csv', index_col=0)
print(data_csv)

"""
   CREATING COPIES OF ORIGINAL DATA
   1. Shallow copy
   2. Deep copy
"""


# shallow copy

copy_data=data_csv
# another method

# any changes in this may be reflected in original copy
shallow_copy=data_csv.copy(deep=False)

deep_copy=data_csv.copy(deep=True) # any changes in this wont be reflected in origia
# -gnal copy

print(deep_copy.index)
print(deep_copy.size)

#print(10770/1795)
print("Here we have shape(row, column)")
print(deep_copy.shape)

print("Below line of code gives memory usaage of each column")
print(deep_copy.memory_usage())


# To get number of axes or array dimension
print(" To get number of axes or array dimension")
print(deep_copy.ndim)

"""
   Indexing and selecting data
   1. By default head() returns first 5 rows
   2. but head(6) returns 6 rows
   
   3. The function tail( ) returns the last n rows for the object based on position
   4. 
"""


print("We want to print first 6 rows of the dta frame")
print(deep_copy.head(6))



print("_____________________________________________________________________________")
print("last n rows of the data frame using tail function")
print(deep_copy.tail(5))

"""

To access the scalar value, the fastest way is to use the "at" and "iat" methods
  1. "at" provides label-based scalar lookups
  2. 
"""

print(deep_copy.at[4, 'Bean Origin'])
# "iat function takes integer based row and column
print(deep_copy.iat[5,5])

"""
  to access a group of rows and columns by label(s) .loc[] can be used
  
"""
print("To access the data of particular row ---> below is the code")  
print(deep_copy.loc[:,['Rating', 'Company Location']])
# or simply we can access multiple column without using .loc[]
print(deep_copy[['Rating', 'Company Location']])

   

