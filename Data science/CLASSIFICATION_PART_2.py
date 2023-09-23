# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:05:52 2023

@author:    SATYA PRAKASH

"""""
"""
   LOGISTIC REGRESSION will start from line no. -- 135
"""
import os
os.chdir("D:/spyder")


import pandas as pd
from bs4 import BeautifulSoup


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
 print(np.unique(data2['occupation']))
"""


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
#sns.displot(data2['age'], bins=10, kde=False)

 ############ BOX PLOT - Age vs Salary Status ##############

#sns.boxplot('SalStat', 'age', data = data2)
data2.groupby('SalStat')['age'].median()



"""
   HERE ON WE WILL DO THE EXPLORATORY DATA ANALYSIS, using the command which i have used in 
   exploratory data analysis
"""


# REINDEXING THE SALARY STATUS names to 0, 1

data2['SalStat'] = data2['SalStat'].map({'less than or equal to to 50, 00'})
print(data2['SalStat'])

new_data = pd.get_dummies(data2, drop_first = True)

# Sorting the column names
columns_list = list(new_data.columns)
print(columns_list)

#separating the input name from data
features = list(set(columns_list) - set(['SalStat']))
print(features)


# Storing the output variables in 'y' 
y=new_data['SalStat'].values
print(y)

# storing value from input features
x = new_data[features]
print(x)

# Splitting data into train and test
train_x,  test_x, train_y,  test_y = train_test_split(x, y, test_size =0.3, random_state = 0)

# make an instance of the model 
logistic = LogisticRegression()

 # fitting value for x and y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_


# prediction from test data

prediction = logistic.predict(test_x)
print(prediction)

# confusion matrix  - - it gives number of correct prediction and incorrect prediction

confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

 # Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y !=prediction).sum())

# ================================================================
# LOGISTIC REGRESSION - REMOVING THE INSIGNIFICANT VARIABLES
# ================================================================

# reindexing the salary status names to 0, 1

data2['SalStat'] = data2['SalStat'].map({'less than or equal to 50,000' : 0,'greater than '})

print(data2['SalStat'])
cols = ['gender', 'nativecountry', 'race', 'JobType']

new_data = data2.drop(cols, axis = 1)

new_data = pd.get_dummies(new_data. drop_first = True)

# Sorting the column names 
columns_list = list(new_data.columns)
print(columns_list)

#separating the input name from data
features = list(set(columns_list) - set(['SalStat']))
print(features)


# Storing the output variables in 'y' 
y=new_data['SalStat'].values
print(y)

# storing value from input features
x = new_data[features]
print(x)

# Splitting data into train and test
train_x,  test_x, train_y,  test_y = train_test_split(x, y, test_size =0.3, random_state = 0)



# Splitting data into train and test
train_x,  test_x, train_y,  test_y = train_test_split(x, y, test_size =0.3, random_state = 0)

# make an instance of the model 
logistic = LogisticRegression()

# fitting value for x and y
logistic.fit(train_x, train_y)

# Prediction from the test data
prediction=logistic.predict(test_x)

# Calculating the accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

# printing the misclassified values from prediction
print('Misclassified samples: %d', %(test_y != prediction).sum())


# ====================================================
# KNN
# ====================================================

# importing library of KNN 

from sklearn.neighbours import kNeighboursClassifier

# import library for plotting
import matplotlib.pyplot as plt



# Storing the K nearest neighbour classifier
KNN_classifier = kNeighboursClassifier(n_neighbours = 5)

# Fitting the values for x and y
KNN_classifier.fit(train_x, train_y)

# Predicting the test values with model
prediction  = KNN_classifier.predict(test_x)

# Performance matrix check
confusion_matrix = confusion_matrix(test_y, prediction)
print("\t", "Predicted Values")
print("Original values ", "\n", confusion_matrix)


# Calculating the accuracy 
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

print('Miscellaneous samples : %d' %(test_y != prediction).sum())

"""
  EFFECT OF K VALUES ON CLASSIFIER
  
"""
Misclassified_samples = []
for i in range(1, 20):
    knn = kNeighboursClassifier(n_neighbours = i)
    knn.fit(train_x,train_y)
    pred_i = knn.predict(test_x)
    Misclassified_samples.append((test_y != pred_i).sum())
    
print(Misclassified_samples)


########################      END   OF   SCRIPT       #########################
















