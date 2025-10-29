import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

k=20
d=2
ants_count = 1000

def probPath(pathOfInterest, pathVector):
    denom = 0
    for i in range(0, len(pathVector)):
        denom = denom + (pathVector[i]+k)**d

    prob = (pathOfInterest + k)**d/denom
    return prob


ant_count_per_path = [0,0]
distance = [1, 2]


def update_ant_count(ants_count_per_path, distance):
    r = np.random.uniform(0,1)
    if r > distance[1]/np.sum(distance):
        ants_count_per_path[1] += 1
    else:
        ant_count_per_path[0] += 1

for i in range(ants_count):
    update_ant_count(ant_count_per_path,distance)

print(ant_count_per_path)