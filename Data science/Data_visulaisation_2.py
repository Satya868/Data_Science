# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 00:59:16 2023

@author: Satya Prakash
"""

"""
  what we will learn 
  WE WILL LEARN HOW TO CREATE BASIC PLOTS USING SEABORN LIBRARY
  
  * Scatter Plot
  * Histogram
  * Bar Plot
  * Box and Whiskers Plot
  * Pairwise Plots

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("D:/spyder")
data_csv = pd.read_csv('flavors_of_cocoa.csv', index_col=0, na_values=["??", "????"])
print(data_csv)

#  sorry I cant do anything as Seaborn is sleeping somewhere

