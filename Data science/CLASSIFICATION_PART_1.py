# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:52:47 2023

@author: SATYA PRAKASH

"""
import os
os.chdir("D:/spyder")



import pandas as pd


import seaborn as sns

from sklearn.moden_selection import train_test_split

from skl.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix


data_csv = pd.read_csv('income.csv')
print(data_csv)

data=data_csv.copy()
print(data.info())
# To check missing values
print(data.isnull())


# Summary of numerical Variables
Summ_data = data.describe()
print(Summ_data)

"""
 SUMMARY FOR CATEGORICAL VARIABLE
 
 
"""

summ_Categorical = data.describe(iclude="0")
print(summ_Categorical)


# ***** Frequency of each CATEGORIES

data['JobType'].value_counts()
data['occupation'].value_counts()

# --------- CHECKING for unique Classes

print(np.unique(data['JobType']))
print(np.unique(data['occupation']))


"""
  GOING BACK and READING THE DATA BY INCLUDING "na_values['?'] to 
"""

d_2 = pd.read_csv('income.csv', na_values=["?"])


"""
   DATA PREPROCESSING
"""

d_2.isnull().sum()
missing = d_2[d_2.isnull().any(axis=1)]


# Here we will remove all the rows with missing Values and we will consider only the 
# rows with values

data2 = d_2.dropna(axis=0)

# Relationship between independent variable
correlation = data2.corr()
print(correlation)


"""
# Fre"quenc"y of "each "categ"ory
"data2['JobType'].value_counts();
data2['occupation'].value_counts();

#Checking for unique classes 
print(np.unique(data2['JobType']))
print(np.unique(data2['occupation']))"""


# Extracting the column name
data2.columns


#Gender proportion Table
gender = pd.crosstab(index = data2["gender"],
                     columns= 'count',
                     normalize=True)

print(gender)


# gender vs salary Status

gender_stat = pd.crosstab(index = data2["gender"],
                     columns= data2['SalStat'],
                     margins=True,
                     normalize='index')

print(gender_stat)

SalStat = sns.countplot(data2['SalStat'])
sns.displot(data2['age'], bins=10, kde=False)

############ BOX PLOT - Age vs Salary Status ##############

sns.boxplot('SalStat', 'age', data = data2)
data2.groupby('SalStat')['age'].median()



"""
  HERE ON WE WILL DO THE EXPLORATORY DATA ANALYSIS, using the command which i have used in 
  exploratory data analysis
"""








