import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os


df = pd.read_fwf('ES_data_7.dat', names=['x','y'])

T=200
mu = 150
lamb = 5*mu


x = df['x']
y = df['y']

vec = np.zeros((len(x),7))

#filling the array with abc and sigma abc
for i in range(len(vec)):
    for j in range(6):
        if j < 3:
            vec[i,j] = random.uniform(-10,10)
        else:
            vec[i,j] = random.uniform(0,10)

#The estimate calculated with the formula taken from the task
def est(a,b,c, i):
    result = a*(x[i]**2 - b * math.cos(c*math.pi*x[i]))
    return result

def err_vec(vector):
    errors = []
    #print(vector)
    for v in vector:
        print(v)
        err = 0


        #error
        for i in range(len(vec)):
            err += (y[i] - est(v[0],v[1],v[2], i))**2

        err = err / np.size(vec)
        print(err)
        errors.append(err)
    return errors

def make_Probs(vector, errors):
    probs = []
    sum_prob = np.sum(errors)
    for p in errors:
        probs.append(p/sum_prob)
    vector[:,6] = probs
    return vector
        

vector2 = make_Probs(vec, err_vec(vec))
print(max(vector2[:,6]))

plt.figure(figsize = (10,6))
plt.scatter(x,y, s = 10) #s is markersize, I think default is 36
plt.xlabel('i')
plt.ylabel('o(i)')
plt.grid(True)
plt.title('True values vs function')
plt.show()

