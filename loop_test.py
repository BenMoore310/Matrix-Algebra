from distutils.log import error
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

dictionary = {}

for n in range(0,5,1):
    
    random_list = np.random.randint(low= -10, high= 10, size = (1,4), dtype= int)

    dictionary.update({n:random_list})

    print(dictionary)
    
arrays = list(dictionary.values())


    


