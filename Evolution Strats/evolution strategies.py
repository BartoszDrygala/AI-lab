import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.random as nprand
import math

random.seed(24)
np.random.seed(24)

df = pd.read_fwf('ES_data_7.dat', names=['x', 'y'])

Time = 200
mu = 150
lamb = mu
n = 3
tau1 = 1 / math.sqrt(2 * n)
tau2 = 1 / math.sqrt(2 * math.sqrt(n))

x = df['x']
y = df['y']

initial_popualiton = np.zeros((mu, 7))

# filling the array with abc and sigma abc
for i in range(mu):
    for j in range(6):
        if j < 3:
            initial_popualiton[i, j] = random.uniform(-10, 10)
        else:
            initial_popualiton[i, j] = random.uniform(0, 10)


# The estimate calculated with the formula taken from the task
def fun(a, b, c, i):
    result = a * (i ** 2 - b * math.cos(c * math.pi * i))
    return result


def err_vec(vector):
    errors = []
    # print(vector)
    for v in vector:
        # print(v)
        err = 0

        # error
        for i in range(len(x)):
            err += (y[i] - fun(v[0], v[1], v[2], x[i])) ** 2

        err = err / len(x)
        # print(err)
        errors.append(err)
    return errors


def make_Probs(vector, errors):
    # Avoid division by zero
    fitness = 1 / (np.array(errors) + 1e-8)
    probs = fitness / np.sum(fitness)
    vector[:, 6] = probs
    return vector


def make_Pool(vector, probs, num):
    pool = []
    for _ in range(num):
        choice = random.choices(vector, weights=probs, k=1)[0]
        pool.append(choice)
    return pool


def make_New_Population(old_pop, size):
    new_population = []
    r1 = nprand.normal(0, 1)
    for i in range(size):

        r2 = nprand.normal(0, 1, 3)
        a = old_pop[i][0] + nprand.normal(0, old_pop[i][3])
        b = old_pop[i][1] + nprand.normal(0, old_pop[i][4])
        c = old_pop[i][2] + nprand.normal(0, old_pop[i][5])

        sigmas = old_pop[i][3:6] * np.exp(tau1 * r1 + tau2 * r2)
        new_population.append(np.array([a, b, c, *sigmas, 0]))
    return new_population


current_population = initial_popualiton
mean_error = np.average(err_vec(current_population))
print(mean_error)
id = 0
#naj potomek od naj rodzica < 10e-5
diff = 1
while diff > 0.000001:
    parents = make_Probs(current_population, err_vec(current_population))
    pool = make_Pool(vector=parents, probs=parents[:, 6], num=mu)

    offspring = make_New_Population(pool, mu)
    combined_population = np.vstack((parents, offspring))
    errors = err_vec(combined_population)
    best_idx = np.argsort(errors)[:mu]
    current_population = combined_population[best_idx]
    best_current = current_population[0]

    errors_par = err_vec(parents)
    best_idx_par = np.argsort(errors_par)[:mu]

    #
    diff = errors[best_idx[0]] - errors_par[best_idx_par[0]]

    mean_error = np.average(err_vec(current_population))
    id = id + 1
    if (id > 200):
        print("too long")
        break
print(id)
errors = err_vec(current_population)
best_idx = np.argsort(errors)
parents = current_population[best_idx]
xs = np.linspace(-5, 5, 150)
ys = []
for _x in xs:
    ys.append(fun(parents[0][0], parents[0][1], parents[0][2], _x))
print(np.average(err_vec(parents)))
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=10)  # s is markersize, I think default is 36
plt.scatter(xs, ys, s=10, c='red')  # s is markersize, I think default is 36
plt.xlabel('i')
plt.ylabel('o(i)')
plt.grid(True)
plt.title('True values vs function')
plt.show()
