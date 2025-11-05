import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
'''
#cities 1
x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
y = [1, 4, 5, 3, 0, 4, 10, 6, 9, 10]'''
k=20
d=2
ants_count = 1000

def probPath(index_of_interest, ants_on_paths, distances):
    weighted_values = []
    for i in range(len(ants_on_paths)):
        weighted_values.append(((ants_on_paths[i] + k) ** d) / distances[i])

    denom = sum(weighted_values)
    prob = weighted_values[index_of_interest] / denom
    return prob

'''dist_matrix = np.zeros((len(x),len(x)))
for i in range(0,len(x)):
    for j in range(0, len(x)):
        dist_matrix[i][j] = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)
    prob = (pathOfInterest + k)**d/denom
    return prob

df = pd.DataFrame(np.array(dist_matrix))'''

'''df.to_excel("twoaj stara.xlsx")'''
ants_count_per_path = [0,0]
distance = [2, 4]


def update_ant_count(ants_count_per_path, distances):
    r = np.random.uniform(0, 1)
    prob0 = probPath(0, ants_count_per_path, distances)
    if r > prob0:
        ants_count_per_path[1] += 1
    else:
        ants_count_per_path[0] += 1

for i in range(ants_count):
    update_ant_count(ants_count_per_path, distance)

print(ants_count_per_path)