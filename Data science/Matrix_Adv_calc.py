# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 23:04:36 2023

@author: Satya Prakash
"""
"""
Aim : Things which we will peform here 
1. create matrix
2. Dimensions
3. Modifying matrices
4. Accessing elements of matrix
5. Mateix operstion

"""
import numpy as np
a=np.matrix("1,2,3,4;4,5,6,7; 7,8,9,10")
print(a); print("(row, col)")
print(a.shape)

# Modifying matrix using insert() 
"""
1. using insert command--- np.insert(matrix,obj==index, values to be inserted, axis)
2. Extract element from 3rd column

"""
print("inserting a new column in the given matrix")
col=np.matrix("11,12,13, 14")
a=np.insert(a, 0, col, axis=0)
print(a)

print("----------")

print("Extracting element from 3rd column")
print(a[:,2])
a=np.transpose(a)
print("transpose of the given matrix")
print(a)

"""
1. Division of Matrix, perform division element wise division between two matrix
syntax is np.divide(m1, m2)
"""
b=np.matrix("20,21,22,23;24,25,26,27;28,29,30,31;32,33,34,35")
print(b.shape)
div_mat=np.divide(a,b)
print(div_mat)


