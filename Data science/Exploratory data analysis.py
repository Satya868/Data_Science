"""
Created on Tue Sep 12 01:25:00 2023

@author: SATYA PRAKASH

"""

import os
import pandas as pd
import numpy as np

os.chdir("D:/spyder")
data_csv = pd.read_csv('flavors_of_cocoa.csv', index_col=0, na_values=["??", "????"])

print(data_csv)
"""
 What we will practice here
 1. Frequency tables
 2. Two-way tables
 3. Two-way tables - joint probability
 4. Two-way tables - marginal probability
 5. Two-way tables - conditional probability
 6. Correlation
 
 
"""
# Creating a Copy of original data
fl_data = data_csv.copy()
print(fl_data)

# Now we will deal with fl_data only

"""
  Creating Frequency Table
  syntax is -- pandas.crosstab() , here dropna == Drop Na 
  @ To compute a simple cross-tabulation of one, two(or more) factors
  @ By default computes a frequency table of factors
  
"""
print("---------------------------------------------------")
print("frequency Table of Rating")

freq_rat=pd.crosstab(index=fl_data['Rating'], columns='count', dropna=True)
print(freq_rat)


print("---------------------------------------------------")
print("frequency Table Of Review year")

freq_yr=pd.crosstab(index=fl_data['Review Date'], columns='count', dropna=True)
print(freq_yr)

"""
   Two way table,  
   
     TO look at the frequency distribution of "Rating" types with respect to different 
     rated in which "Year"
"""


print("---------------------------------------------------")
print("Print of two way data with Review date and rating")

two_fq=pd.crosstab(index=fl_data['Rating'], columns=fl_data['Review Date'], dropna=True)
print(two_fq)


"""
Two-way table -- joint probability

@ JOINT PROBABILITY -- It is likelihood of two independent events happening at 
          the same time.
       
     HOW TO DO IT?
     =====--- we just add "normalise = True," - it convert all the table from 
              numbers to proportion.     

"""


print("---------------------------------------------------")
print("Print of two way data and its joint probability")

jt_pb=pd.crosstab(index=fl_data['Rating'], columns=fl_data['Review Date'], normalize=True, dropna=True)
print(jt_pb)

"""
   MARGINAL PROBABILITY-- It is probability of occurance of single event
"""
print("---------------------------------------------------")
print("Print of two way data and its marginal probability")

mg_pb=pd.crosstab(index=fl_data['Rating'], columns=fl_data['Review Date'],margins=True, normalize=True, dropna=True)
print(mg_pb)


"""
   Conditional Probability:-> it is orobability of an event(A), given that 
   another event (B) has already occured.
       given the type of rating , probability of different year.
       
   here the row sum comes out to be 1.
"""

print("---------------------------------------------------")
print("Two way data and its conditional probability")

cnd_pb=pd.crosstab(index=fl_data['Rating'], columns=fl_data['Review Date'],margins=True, normalize='index', dropna=True)
print(cnd_pb); 

"""
  Another case.... LOL i didnt understood the difference between these two yet 
"""
print("---------------------------------------------------")
print("Two way data and its conditional probability")

cnd_pb_1=pd.crosstab(index=fl_data['Rating'], columns=fl_data['Review Date'],margins=True, normalize='columns', dropna=True)
print(cnd_pb_1); 


"""
    CORRELATION -- To check the strength between two variables
    
    Can be classified into 3 categories
    1. Positive trend(Positive Slope AT PI/4)
    2. Negative Trend(Negative slope)
    3. Little or No correlation(Spreaded all over the  Plot)
    
    SYNTAX -- 
    dataframe.corr(self, method='pearson'){
          1. to compute pair wise correlation of columns excluding NA/null values
          2. Excluding the categorical variables to find the Pearson's correlation
        
        }
"""

num_data=fl_data.select_dtypes(exclude=[object])
print("Correlation of data")
print(num_data)
print(num_data.shape)
print(num_data.corr())







