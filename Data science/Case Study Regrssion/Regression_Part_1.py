# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:34:49 2023

@author: SATYA PRAKASH

PROBLEM STATEMENT: Predicting price of Pre-owned Cars

"""

"""
                      ####   DESCRIPTION   ####
                      
      AIM: To develop an algorithm to predict the price of the cars based on various
         attributes associated with the car
"""

import pandas as pd
import numpy as np
import seaborn as sns

# fucking shit it worked now(Seaborn)

# ================================================================
#      SETTING DIMENSION FOR PLOT
# ================================================================

sns.set(rc = {'figure.figsize' :(11.7, 8.27)})


# ================================================================
#     READING CSV FILE
# ================================================================

cars_data=pd.read_csv('cars_sampled.csv')
#print(cars_data)

# ================================================================
#     CEATING COPY
# ================================================================
cars = cars_data.copy()

#print(cars.info())

# ================================================================
#     SUMMARIZEE THE DATA
# ================================================================

#desc = (cars.describe())

# Below function is used to convert the scientific notation into Simple variable type
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

desc = (cars.describe())
print(desc)



# ================================================================
#     TO DISPLAY MAXIMUM SET OF COLUMNS
# ================================================================
pd.set_option('display.max_columns', 500)

desc = (cars.describe())
print(desc)


# ================================================================
#     DROPPING UNWANTED COLUMN
# ================================================================

"""
 WE will be dropping the column name such as == 'name', 'dateCrawled', 'postalCode', and 'lastseen'
 
"""
col = ['name', 'dateCrawled', 'postalCode','lastSeen']
cars = cars.drop(columns=col, axis =1)
print("Here is car data after Dropping unwanted column")
print(cars)



# ================================================================
#     REMOVING DUPLICATE RECORDS
# ================================================================ 
cars.drop_duplicates(keep='first', inplace=True)
#470 columns are duplicate


# ================================================================
#        DATA CLEANING
# ================================================================ 

print( \
""" ================================================================ """ \
"""                                                   DATA CLEANING                                    """ \
""" ================================================================   """ \
)

#    No. of missing values is each column
print("NUmber of missing values in each columns ")
print(cars.isnull().sum())

# Variable -- yearOfRegistration

yr_wise_count = cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration'] > 2018)
sum(cars['yearOfRegistration'] < 1950)
sns.regplot(x = 'yearOfRegistration', y='price', scatter=True, fit_reg = False, data=cars)


# working range - 1950 and 2018



# variable name - price
price_count  =cars['price'].value_counts().sort_index()
sns.displot(cars['price'])
cars['price'].describe()
sns.boxplot(y = cars['price'])
sum(cars['price'] > 15000)
sum(cars['price'] < 100)
# working range 100 and 15000

# variable -- powerPS
pow_count = cars['powerPS'].value_counts().sort_index()
print(pow_count)
sns.displot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y = cars['powerPS'])
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)
sum(cars['powerPS'] > 500)
sum(cars['powerPS'] < 10)
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)

#   working range 10 and 500



# ================================================================
#        WORKING RANGE OF DATA
# ================================================================ 

cars = cars[(cars.yearOfRegistration <= 2018) & (cars.yearOfRegistration >= 1950) &(cars.price >= 100) &(cars.price >= 150000) &(cars.powerPS >= 10) &(cars.powerPS <= 500)]

# dropped 6700 records

# Further to simplify - variable reduction
# Combining year of registration and monthOfRegistration

cars['monthOfRegistration']/=12

# creating new variable Age by adding yearOfRegistration and monthOfRegistration
cars['Age'] = (2018+cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age'] = round(cars['Age'], 2)
cars['Age'].describe()

# Dropping yearOfRegistration and monthOfRegistration
cars = cars.drop(columns = ['yearOfRegistration', 'monthOfRegistration'], axis=1) 

print(cars)
# Visualizing Parameters

# 1. AGE

sns.displot(cars['Age'])
sns.boxplot(y=cars['Age'])


# 2. price
sns.displot(cars['price'])
sns.boxplot(y = cars['price'])

# 3. powerPS
sns.displot(cars['powerPS'])
sns.boxplot(y = cars['price'])

# 4.    Visualizing parameters after narrowing working Range
#   Age vs Price

sns.regplot(x='Age', y='price', scatter=True, fit_reg=False, data=cars_data)
# car priced higher are fewer
# With increase in age, price decreases
# However some cars are priced higher with increase in age


# powerPS vs price
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)


# Variable Seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'], columns='count', normalize=True)
sns.countplot(x = 'seller', data=cars)

# fewer cars have 'commercial' => Insignificant

# Variable OffeType
cars['offerType'].value_counts()
pd.crosstab(cars['offerType'], columns='fdg', normalize=True)
sns.counterplot(x= 'offerType', data=cars)

# fewer cars have 'offer' => Insignificant


# Variable obtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'], columns='count', normalize=True)
sns.counterplot(x= 'obtest', data=cars)

# Equally Distributed

sns.boxplot(x='abtest', y='price', data=cars)


# For every price value there is almost 50-50 
# Does not affect price => Insignificant



# Variable  vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'], columns='count', normalize=True)
sns.counterplot(x='vehicleType', data=cars)
sns.boxplot(x='vehicleType', y='price', data=cars)

# Variable GearBox

cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'], columns='count', normalize=True)
sns.countplot(x='gearbox', data=cars)
sns.boxplot(x='gearbox', y='price', data=cars)

# Variable -- kilometer
cars['kilometer'].value_counts()
pd.crosstab(cars['kilometer'], columns='count', normalize=True)
sns.boxplot(x='kilometer', y='price', data=cars)
cars['kilometer'].describe()
sns.displot(cars['kilometer'], bins=8, kde=False)
sns.regplot(x='kilometer', y='price', scatter=True, fit_reg=False, data=cars)


# fuel Type affects price
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'], columns='count', normalize=True)
sns.countplot(x='fuelType', data=cars)
sns.boxplot(x='fuelType', y='price', data=cars)

# Variable brand 
cars['brand'].value_counts()
pd.crosstab(cars['brand'], columns='count', normalize=True)
sns.countplot(x='brand', data=cars)
sns.boxplot(x='brand', y='price', data=cars)

# Cars are distributed over many brands
 # considered for modelling
# Yes - car is damaged but not ratified 
# No- car was damaged but has been ratified

cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'], columns='count', normalize=True)
sns.countplot(x='notRepairedDamage', data=cars)
sns.boxplot(x='notRepairedDamage', y='price', data=cars)

# As expected the car that require the damages to be repaired



# ================================================================
#       REMOVING INSIGNIFICANT VARIABLE
# ================================================================ 
col=['seller','offerType', 'abtest']
cars_copy = cars.copy()


# ================================================================
#       CORRELATION
# ================================================================ 
cars_select1 = cars.select_dtypes(exclude=[object])
print(cars_select1)
correlation =cars_select1.corr()
round(correlation, 3)
cars_select1.corr().loc[:, 'price'].abs().sort_values(ascending=False)[1:]


# ================================================================
#        OPTIMISING MISSING VALUES
# ================================================================ 

# =========================

"""
     We are going to build a Linear Regression and Random Forest MOdel
     on two data sets
     
     1. Data obtained by omitting rows with any missing values
     2. Data obtained by inputting the missing values
     
"""

cars_omit = cars.dropna(axis=0)

# converting categorical variables to dummy variables

cars_omit= pd.get_dummies(cars_omit, drop_first=True)

"""
         IMPORTING NECESSARY LIBRARIES
"""     

#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



# ================================================================
#       MODEL BUILDING WITH OMITTED DATA
# ================================================================ 

# Separasting input and output features

x1 = cars_omit.drop(['price'], axis='columns', inplace=False)
y1 = cars_omit['price']

# Plotting the variable price
prices = pd.DataFrame({"1. Before": y1, "2. After": np.log(y1)})
prices.hist()


# Transforming price as a logarithmic value

y1 = np.log(y1)

# Splitting Data into train and test
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
#x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size =0.3, random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)




# ================================================================
#       BASELINE MODEL FOR OMITTED DATA
# ================================================================ 

"""
  We are making A BASE model by using test data mean value
  
  This is to set a benchmark and to compare with our regression model 
"""

# Finding the mean for test data value
base_p = np.mean(y_test)
print(base_p)

#Repeating some of the value till length of test data
base_p = np.repeat(base_p, len(y_test))


# finding RMSE(Root Mean Square Error) value
rms_error = np.sqrt(mean_squared_error(y_test, base_p))
print(rms_error)
"""
 PURPOSE OF CALCULATING THIS  -- THIS IS BENCHMARK, ANY FUTURE VALUE SHOULD BE LESS THAN THIS
"""


# ================================================================
#       LINEAR REGRESSION WITH CERTIFIED DATA
# ================================================================ 

# Setting intercept as true
lgr = LinearRegression(fit_intercept=True)

#Model
model_linear  = lgr.fit(X_train, y_train)


# Predicting Model on test set
cars_pred_li = lgr.predict(X_test)


# Computing MSE(mean square error) and RMSE
lin_mse = mean_squared_error(y_test, cars_pred_li)
lin_rmse = np.squt(lin_mse)
print(lin_rmse)


# R squared value
r2_lin_test1 = model_linear.score(X_test, y_test)
r2_lin_train2 = model_linear.score(X_train, y_train)
print(r2_lin_test1, r2_lin_train2); # It will print the value using which you can guess whether model is good or bad


# Regression Diagnostic : Residual Plot analysis
residual = y_test.cars_pred_li
sns.regplot(x=cars_pred_li, year=residual, scatter=True, fit_reg=False, data=cars)
residual.describe()




# ================================================================
#       RANDOM FOREST WITH OMITTED DATA
# ================================================================ 

rf =RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=100, min_samples_split=10,
                          min_samples_leaf=4, random_state=1)


#MODEL
model_rf1 = rf.fit(X_train, y_train)

# Predicting Model on train set
car_pred_rf1 = rf.predict(X_test)

# Connecting MSE and RMSE

rf_mse = mean_squared_error(y_test, car_pred_rf1)
rf_rmse = np.sqrt(rf_mse)

print(rf_rmse)


# R squared value
r2_rf_tst = model_rf1.score(X_test, y_test)
r2_rf_train = model_rf1.score(X_train, y_train)
print(r2_rf_tst,r2_rf_train)


# ================================================================
#       MODEL BUILDING WITH IMPUTED DATA
# ================================================================ 

cars_imputed = cars.apply(lambda x: x.fillna(x.median()) \
                          if x.dtype == 'float' else \
                              x.fillna(x.value_counts().index[0]))

cars_imputed.isnull().sum()

# Converting categorical variable into dummy variable

cars_imputed = pd.get_dummies(cars_imputed, drop_first=True)



# Separating input and Output features
x2 = cars_imputed.drop(['price'], axis='column', inplace=True)
y2 = cars_imputed['price']


# Plotting the variable price
prices = pd.DataFrame({"1. Before": y2, "2. After": np.log(y2)})
price.hist()


# Transforming price as a logarithmic value
y2 = np.log(y2)


X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(x2, y2, test_size=0.3, random_state = 3)
#x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size =0.3, random_state=3)
print(X_train_1.shape, X_test_1.shape, y_train_1.shape, y_test_1.shape)




# ================================================================
#       BASELINE MODEL FOR IMPUTED DATA
# ================================================================ 





















