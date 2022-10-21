from distutils.log import error
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit


# Script iterates through series of n x n matrices for n in range 1 to 100, checks via calculation of submatrices determinants whether 
# LU decomposition is possible. If so, performs LU decomposition, compares final value to target value from numpy library function.
# if LU decomposition is not possible for matrix, matrix is regenerated. Time taken to perform LU decomposition for each matrix is 
# recorded and plotted against matrix size, along with a fitted curve to investigate size vs time relationship.


# initialisation of matrix size and time matrices to store relevant data for each matrix
size_values = []  
time_values = []

# initialisation of for loop to iterate through range of matrix sizes
for size in range(1,101,1):
    bad_det = True 
    while bad_det == True:
        det_error = False
        matrix_size = int(size)  #put in a limiter on matrix size
        print('Generating your random {}x{} matrix...'.format(matrix_size,matrix_size))
        random_matrix = np.random.randint(low= -100, high= 100, size = (matrix_size, matrix_size), dtype= int)
        # print(random_matrix)
        m = random_matrix

        #need a block that calculates the determinants of leading sub matrices to check whether they have a zero determinant
        print('Calculating determinants of leading sub-matrices...')
        #loop iterates through all submatrices of generated matrix, sets det_error to true if any submatrices determinants = 0 so matrix
        # can be regenerated.
        for i in range(1, matrix_size+1):
            submatrix = random_matrix[0:i, 0:i]
            # print('\n Sub-matrix {}:\n'.format(i),submatrix)
            sub_det = np.linalg.det(submatrix)  #submatrix determinants calculated using numpy library
            # print('\n Sub-matrix {} determinant = '.format(i),sub_det, '\n')
            if sub_det == 0:
                print('LU decomposition will not work for this matrix')
                det_error = True 

        if det_error == True:
            bad_det = True
            print('LU decomposition will not work for this matrix, regenerating matrix... \n')
        
        #if no submatrix determinant = 0, LU decomposition is possible and script continues
        else:
            bad_det = False
            print('LU decomposition is possible for this matrix \n')


    # special case check - brute force method for a 3x3 matrix only
    if matrix_size == 3:
        bdet = (m[0,0]*(m[1,1]*m[2,2]-m[2,1]*m[1,2])-(m[0,1]*(m[1,0]*m[2,2]-m[2,0]*m[1,2]))+(m[0,2]*(m[1,0]*m[2,1]-m[2,0]*m[1,1])))
        print('Brute force target value for 3x3 determinant:', bdet)
    
    #all other cases - target determinant value calculated using numpy library
    print('Numpy library target value for determinant:', np.linalg.det(m))

    # beginning of LU decomposition


    size_values.append(matrix_size)
    startTime = time.time()

    #initializing empty u and l matrices
    u =np.zeros((matrix_size, matrix_size))
    l =np.zeros((matrix_size, matrix_size))
    n = matrix_size

    for j in range(0, matrix_size): # calculating first row of u
        u[0,j] = m[0,j]
    #print(u)   


    for j in range(0, matrix_size):  #calculting first collumn of l
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

    #product of diagonal elements in u matrix = determinant of original matrix
    final_sum = 1
    for i in range(0, n):
        value = u[i][i]
        final_sum = final_sum * value


    print('\nCalculated determinant =', final_sum)
    endtime = time.time() #time to perform decomposition recorded
    total_time = endtime- startTime
    print('\nCalculation completed in',total_time, 'seconds.')
    time_values.append(total_time) 

    # print(size_values, time_values)

#defining target function for curve fitting of data
def test_function(x,a,b):
    return a * (x**b)

coeffs, coeffs_covarience = curve_fit(test_function, size_values, time_values)
fit = coeffs[0]*(size_values**coeffs[1])
# print(coeffs)
plt.plot(size_values, time_values, label = 'raw data')
plt.plot(size_values, fit, label = 'Curve fit ax^b, power = {}'.format(coeffs[1]), alpha = 0.7)
plt.xlabel('Number of rows in matrix')
plt.ylabel('Time taken to complete LU decomposition (s)')
plt.title('Time taken vs Matrix size')
# plt.text(0,0.20, {'${ax^b}$, a = ',coeffs[0],' , b = ', coeffs[1]}, fontsize = 8)
plt.legend()
plt.show()

