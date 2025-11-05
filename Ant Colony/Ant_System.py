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
#print(pheromone_matrix)

decision_table = np.zeros(shape)
#print(decision_table)

for i in range(n):
    for j in range(n):
        if i != j:
            numer = (pheromone_matrix[i][j] ** alpha) * ((1 / distances[i][j]) ** beta)
            denom = 0
            for k in range(n):
                if k != i:
                    denom += (pheromone_matrix[i][k] ** alpha) * ((1 / distances[i][k]) ** beta)
            denom /= 2
            decision_table[i][j] = numer / denom

probs = decision_table/np.sum(decision_table)

print(probs)


def update_pheromones(pheromone_matrix, ants_paths, distances, rho):
    """
    Update pheromone matrix based on ants' tours.

    Parameters:
        pheromone_matrix (ndarray): current pheromone levels (n x n)
        ants_paths (list of lists): each ant's tour, e.g. [[0, 2, 4, 1, ...], [...], ...]
        distances (ndarray or DataFrame): matrix of distances between cities
        rho (float): pheromone evaporation rate (0 < rho < 1)
    """
    n = len(pheromone_matrix)
    m = len(ants_paths)

    # Evaporation step
    pheromone_matrix *= (1 - rho)

    # Deposit pheromones
    for path in ants_paths:
        # Compute total tour length for this ant
        Lk = 0
        for i in range(len(path) - 1):
            Lk += distances[path[i]][path[i + 1]]
        # add return to start to make it a complete tour
        Lk += distances[path[-1]][path[0]]

        # Deposit pheromone on all edges in this tour
        delta_tau = 1 / Lk
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            pheromone_matrix[a][b] += delta_tau
            pheromone_matrix[b][a] += delta_tau  # because graph is symmetric
        # Also reinforce the closing edge
        pheromone_matrix[path[-1]][path[0]] += delta_tau
        pheromone_matrix[path[0]][path[-1]] += delta_tau

    return pheromone_matrix

for z in range(200):
    update_pheromones(pheromone_matrix, )


ants_count = 1000

def probPath(index_of_interest, ants_on_paths, distances):
    weighted_values = []
    for i in range(len(ants_on_paths)):
        weighted_values.append(((ants_on_paths[i] + k) ** d) / distances[i])

    denom = sum(weighted_values)
    prob = weighted_values[index_of_interest] / denom
    return prob

