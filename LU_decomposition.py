import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime


#loop through function for values of n 2-100, append lists containing n and time taken to find lu decomposition

bad_det = False
print('Welcome, what size do you want your matrix?')
matrix_size = input()
matrix_size = int(matrix_size)  #put in a limiter on matrix size
print('Generating your random {}x{} matrix...'.format(matrix_size,matrix_size))
random_matrix = np.random.randint(low= -10, high= 10, size = (matrix_size, matrix_size), dtype= int)
#regenerate is first element is zero
print(random_matrix)
m = random_matrix

#need a block that calculates the determinants of leading sub matrices to check whether they have a zero determinant
print('Calculating determinants of leading sub-matrices...')
for i in range(1, matrix_size+1):
    submatrix = random_matrix[0:i, 0:i]
    print('\n Sub-matrix {}:\n'.format(i),submatrix)
    sub_det = np.linalg.det(submatrix) #check whether using numpy for this is allowed
    print('\n Sub-matrix {} determinant = '.format(i),sub_det, '\n')
    if sub_det == 0:
        print('LU decomposition will not work for this matrix')
        bad_det = True

if bad_det == True:
    print('LU decomposition will not work for this matrix')
    quit()
else:
    print('LU decomposition is possible for this matrix \n')

#verify whether LU decomposition of particular matrix is possible

#brute force for 3x3 matrix
if matrix_size == 3:
    bdet = (m[0,0]*(m[1,1]*m[2,2]-m[2,1]*m[1,2])-(m[0,1]*(m[1,0]*m[2,2]-m[2,0]*m[1,2]))+(m[0,2]*(m[1,0]*m[2,1]-m[2,0]*m[1,1])))
    print('brute force:', bdet)
print('Target value for determinant:', np.linalg.det(m))

#LU composition
startTime = datetime.now()

#initializing u and l matrices
u =np.zeros((matrix_size, matrix_size))
l =np.zeros((matrix_size, matrix_size))
n = matrix_size

for j in range(0, matrix_size): #first row of u
    u[0,j] = m[0,j]
#print(u)   


for j in range(0, matrix_size):  #first collumn of l
    l[j,0] = (m[j,0])/(u[0,0])
    l[j][j] = 1

#print(l)

for i in range(1,matrix_size-1):
    #print(i)
    j=i+1
    
    if i == 1:  #u diagonal in case where i = 1 (2nd main diag element)
        t=0
        u[i][i] = m[i][i] - ((l[i][t])*(u[t][i]))
    else: #all other cases of u element on main diagonal
        initial_value_1 = m[i][i]
        value_1 = 0
        for t in range(0, i):
            summation_1 = ((l[i][t])*(u[t][i]))
            value_1 += summation_1
        u[i][i] = initial_value_1 - value_1

    for j in range(i+1, n):
        initial_value_2 = m[i][j] #values of u off the main diagonal
        # print(initial_value_2)
        value_2 = 0
        for t in range(0, i):
            # print(t)
            summation_2 = ((l[i][t])*(u[t][j]))
            value_2 += summation_2
            # print(value_2)
        u[i][j] = initial_value_2 - value_2

    #values of l below main diagonal
    for j in range(i+1, n):
        initial_value_3 = m[j][i]
        value_3 = 0
        for t in range(0, i):
            summation_3 = ((l[j][t])*(u[t][i]))
            value_3 += summation_3
        l[j][i] = (initial_value_3-value_3)/u[i][i]

    # print(u)
    # print(l)

#final value m[n][n] of matrix
initial_value_4 = m[n-1][n-1]
value_4 = 0
for t in range(0, n-1):
    summation_4 = ((l[n-1][t])*(u[t][n-1]))
    value_4 += summation_4
    #print('value_4=', value_4)
u[n-1][n-1] = initial_value_4 - value_4    


# print(u)
# print(l)

final_sum = 1
for i in range(0, n):
    value = u[i][i]
    final_sum = final_sum * value

print('\nCalculated determinant =', final_sum)
print('\nCalculation completed in', datetime.now() - startTime, 'seconds.')

