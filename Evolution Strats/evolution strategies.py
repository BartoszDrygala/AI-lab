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
    # Avoid division by zero
    fitness = 1 / (np.array(errors) + 1e-8)
    probs = fitness / np.sum(fitness)
    vector[:,6] = probs
    return vector

def make_Pool(vector, probs, num):
    pool = []
    for _ in range(num):
        choice = random.choices(vector, weights=probs, k=1)[0]
        pool.append(choice)
    return pool



vector2 = make_Probs(vec, err_vec(vec))
probs = vector2[:,6]
print(len(probs), len(vector2))
pool = make_Pool(vector=vector2, probs=probs, num=len(x))
print(pool)
print(type(pool))
print(type(pool[0]))
print(type(pool[0][0]))

'''plt.figure(figsize = (10,6))
plt.scatter(x,y, s = 10) #s is markersize, I think default is 36
plt.xlabel('i')
plt.ylabel('o(i)')
plt.grid(True)
plt.title('True values vs function')
plt.show()'''

