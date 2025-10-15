import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

#header=none is there because this function took first line of data as header
df = pd.read_csv('ES_data_7.dat', sep=r'\s+',header = None)

x = df.iloc[:,0].to_numpy()
y = df.iloc[:,1].to_numpy()

vec = np.zeros((len(x),6))

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

err = 0

#error
for i in range(len(vec)):
    err += (y[i] - est(vec[i, 1],vec[i, 2],vec[i, 3], i))**2

err = err / np.size(vec)

print(err)


plt.figure(figsize = (10,6))
plt.scatter(x,y, s = 10) #s is markersize, I think default is 36
plt.xlabel('i')
plt.ylabel('o(i)')
plt.grid(True)
plt.title('True values vs function')
plt.show()

