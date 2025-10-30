import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

k=20
d=2
ants_count = 1000
ant_count_per_path = [0,0]
distance = [1, 2]

def probPath(path_of_interest, path_vector, distance):
    """
    Compute probability of choosing a given path, considering pheromone and distance.
    """
    pheromone_term = np.array([(path_vector[i] + k) ** d for i in range(len(path_vector))])
    distance_term = np.array([1 / distance[i] for i in range(len(distance))])  # inverse distance
    combined = pheromone_term * distance_term
    denom = np.sum(combined)
    prob = (path_vector[path_of_interest] + k) ** d * (1 / distance[path_of_interest]) / denom
    return prob





def update_ant_count(ant_count_per_path, distance):
    r = np.random.uniform(0,1)
    prob = probPath(0, ant_count_per_path,distance)

    if r > prob:
        ant_count_per_path[1] += 1
    else:
        ant_count_per_path[0] += 1

for i in range(ants_count):
    update_ant_count(ant_count_per_path,distance)

print("Final ant counts per path:", ant_count_per_path)
print("Total ants:", np.sum(ant_count_per_path))
print("Path 0 (shorter) percentage:", ant_count_per_path[0] / ants_count * 100, "%")
print("Path 1 (longer) percentage:", ant_count_per_path[1] / ants_count * 100, "%")