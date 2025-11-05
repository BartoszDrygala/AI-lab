import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import make_xl

filename = "Antek Mrowka.xlsx"
#cities 4
x = [3, 2, 12, 7,  9,  3, 16, 11, 9, 2]
y = [1, 4, 2, 4.5, 9, 1.5, 11, 8, 10, 7]

#params
k=20
d=2
m_cities = len(x)
n_ants = m_cities
alpha = 1
beta = 5
rho = 0.5
init_phero = 6.7

#make_xl.make_xl(filename, x, y)

distances = pd.read_excel(filename, index_col=0)
#print(distnaces)

shape = np.shape(distances)
n = len(distances)

pheromone_matrix = np.full(shape, init_phero)
np.fill_diagonal(pheromone_matrix, 0)
print(pheromone_matrix)




ants_count = 1000

def probPath(index_of_interest, ants_on_paths, distances):
    weighted_values = []
    for i in range(len(ants_on_paths)):
        weighted_values.append(((ants_on_paths[i] + k) ** d) / distances[i])

    denom = sum(weighted_values)
    prob = weighted_values[index_of_interest] / denom
    return prob

