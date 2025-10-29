import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#cities 1
x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
y = [1, 4, 5, 3, 0, 4, 10, 6, 9, 10]


dist_matrix = np.zeros((len(x),len(x)))
for i in range(0,len(x)):
    for j in range(0, len(x)):
        dist_matrix[i][j] = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)

df = pd.DataFrame(np.array(dist_matrix))

df.to_excel("twoaj stara.xlsx")
