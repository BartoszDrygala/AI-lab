import numpy as np

ants_count_per_path = [0,0]
distance = [2, 4]
ants_count = 1000

def probPath(index_of_interest, ants_on_paths, distances):
    weighted_values = []
    for i in range(len(ants_on_paths)):
        weighted_values.append(((ants_on_paths[i] + k) ** d) / distances[i])

    denom = sum(weighted_values)
    prob = weighted_values[index_of_interest] / denom
    return prob

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