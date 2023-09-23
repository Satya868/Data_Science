# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:25:58 2023

@author: SATYA PRAKASH
"""
import os
import pandas as pd
os.chdir("D:/spyder")
data_csv = pd.read_csv('flavors_of_cocoa.csv', index_col=0)
print(data_csv)




"""

import pandas as pd

# Replace the URL with the actual URL of the CSV file you downloaded
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Define column names for the dataset
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Load the CSV into a Pandas DataFrame
df = pd.read_csv(url, header=None, names=column_names)

# Display the first few rows of the dataset
print(df.head())
"""

