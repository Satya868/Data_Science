# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:52:50 2023

@author: SATYA PRAKASH

"""
"""

 WHAT WE WILL LEARN HERE
   1.  We learn how to create basic plots using MATPLOTLIB library
      a.) Scatter Plot
      b.) Histogram
      c.) bar plot
   2. Data visulaisation allows us to quickly interpret the data and adjust different 
      variables to see their effect
   3. Technology is incresingly making it easier for us to do so
   
   
   BUT WHY?
   1. observe the pattern
   2. Identify extreme values that could be anomalies
   3. easy Interpretation
   
   WHAT WE WILL USE
   1. MATPLOTLIB - Matplotlib is a 2D plotting library which produces good quality figures
                 - It is independent of Matlab
                 -It makes heavy use of NumPy and other extension code to provide good 
                   performance even for large array
                  
   2. SCATTER PLOT - A scatter plot is a set of points that represents the values obtained for two
                  different variables plotted on a horizontal and vertical axes
                  
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir("D:/spyder")
data_csv = pd.read_csv('flavors_of_cocoa.csv', index_col=0, na_values=["??", "????"])

# removing missing values ---  causes the error

#data_csv.dropna(axis=0, inplace=True)

print(data_csv)


x_column = 'Cocoa Percent'  # Replace with the appropriate column name
y_column = 'Rating'        # Replace with the appropriate column name

# Create the scatter plot
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.title(f'Scatter Plot of {x_column} vs. {y_column}')

# Display the plot

plt.scatter(data_csv[x_column], data_csv[y_column], c='red')
plt.show()


"""
 Creating Histogram
   * it Is graphical representation of data using bars of different heights
   * Histograms groups numbers into ranges and the height of each bar depicts the frequency
      of each range
   WHEN TO USE HISTOGRAMS
   * To represent the frequency distribution of numerical variables   
"""


plt.title('Histogram of rating')
plt.xlabel('Ratings')
plt.ylabel('Number of company')
plt.hist(data_csv['Rating'])
plt.show()


"""
  Bar Plot
  * A bar plot is plot that presents categorical  data with rectangular bars 
    with lengths proportional to the counts that they represents
    
    
  WHEN TO USE BAR PLOT
  * To represent the frequency distribution of categorical variables
  * a bar diagram makes it easy to compare sets of data between different groups
  
"""

counts =[3298, 730, 2000]
types_of_student = ('Good', 'Bad', 'Average')
index = np.arange(len(types_of_student))
plt.bar(index, counts, color=['red', 'blue', 'cyan']);
plt.title("Bar plot of students")

# we can give label to the bars using x-ticks
plt.xticks(index, types_of_student, rotation=45)

plt.xlabel('students')
plt.ylabel('Frequency')
plt.show()











