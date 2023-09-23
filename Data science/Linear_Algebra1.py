# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 00:14:42 2023

@author: SATYA PRAKASH
"""

"""
Some of the operation which we will see here 

1. Determinant of Matrix
2. Rank of Matrix
3.  Inverse of Matrix
4. Solving system of equation
"""
import numpy as np

a=np.matrix("10,2,3;4,5,6;7,8,9")
print(a)
# syntax of deteminant of matrix
det=np.linalg.det(a)
print("Deteminamt of matrix")
print(det)

#rank ==
print("rank of given matrix :-->", np.linalg.matrix_rank(a))
#print(np.linalg.matrix_rank(a))

print("Inverse of the given above matrix:-->")
print (np.linalg.inv(a))

"""
System of Linear Equation
so here we have three equations
   3x + y + 2z = 2
   3x + 2y + 5z = -1
   6x + 7y + 8z = 3
   
solution --- 
     AX = B
     
     one line code is np.linalg.solve(a,b)
     

"""
A=np.matrix("3, 1, 2; 3, 2, 5; 6, 7, 8 ")
B=np.matrix("2, -1, 3").transpose()
sol=np.linalg.solve(A, B)
print(sol)
print("value of x:", sol[0][0], "value of y:", sol[1][0], "value of z:",sol[2][0])



