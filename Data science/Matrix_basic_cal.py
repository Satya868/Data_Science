import numpy as np
a=np.array([[1,2,3], [4,5,6], [7,8,9]])
b=np.array([[74,9,4],[2,4,5],[98,78,32]])

c=np.add(a,b); print(c); print("..........")

#printing array ad
print("below is first array")
print(a)
print("..........")
d=np.multiply(a,b); print(d)

print("..........")


# for below command both are same with same output
print(a[0][2]); print(a[0,2])


print("..........")

print(a[1:2])

print("..........")
print("..........")


# printing first row
print("first row--")
print(a[0,:])

print("..........")
# printing a subset of 2X2 martrix
print("printing a subset of 2X2 matrix")
# a[0,0]=2
# print(a)
print("transpose of a matrix")
print(np.transpose(a))


print("-------------")
#adding a new array along the row wise
print("adding a new array along the row wise")
new_arr=np.append(a,[[10,11,12]], axis=0)
print(new_arr)


print("----------")
#Adding new array along the coloumn wise
#   for this we will folow the certain command
#   basically steps and the steps are
#   1.. create a array and transform it in column dirction using reshape command
#   2... now add that array in the existing array.


# here goes the code for the addition of new array into existing array leading it to modification
print("Adding new array along the coloumn wise")
col=np.array([[13,14,15, 16]]).reshape(4,1);

Anew_arr=np.append(new_arr, col, axis=1)
# a=Anew_arr
print(Anew_arr)
print("done")

#modifying array using insert() function
# we will be doing this operation with the array name Anew_arr
a_ins=np.insert(Anew_arr, 1, [17, 18, 19, 20], axis=1)
print("Insertion of element along the index position ")
print(a_ins)

#deleting 3rd row from the matrix a_ins ---here axis=0 specify the along row
# and axis =1 specify along the column
a_del = np.delete(a_ins, 2, axis=0)
print("MAtrix after deleting one row at index of === 3")
print (a_del)


#print(col)
# print (a); print(a.shape)
# print("Hello, World!")