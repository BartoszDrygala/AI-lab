import numpy as np
import math
import numpy.random as nprand
def fun(a, b, c, i):
    return a * (i**2 - b * math.cos(c * math.pi * i))


def err_vec(vectors, x, y):
    errors = []
    for v in vectors:
        pred = [fun(v[0], v[1], v[2], xi) for xi in x]
        mse = np.mean((y - pred) ** 2)
        errors.append(mse)
    return np.array(errors)

#Probability assignment (roulette-wheel)
def make_Probs(errors):
    fitness = 1 / (errors + 1e-8)
    return fitness / np.sum(fitness)

# Parent selection
def make_Pool(population, probs, num):
    indices = np.random.choice(len(population), size=num, p=probs)
    return population[indices].copy()

# --- Mutation ---
def make_New_Population(old_pop, size, tau1, tau2):
    new_population = []
    r1 = nprand.normal(0, 1)
    for i in range(size):
        
        r2 = nprand.normal(0, 1, 3)

        # Adaptive step size — allows exploration
        '''sigmas = np.maximum(
            old_pop[i][3:6] * np.exp(tau1 * r1 + tau2 * r2),
            1e-3  # prevents vanishing sigmas
        )'''

        sigmas = old_pop[i][3:6] * np.exp(tau1 * r1 + tau2 * r2)
        a = old_pop[i][0] + nprand.normal(0, sigmas[0])
        b = old_pop[i][1] + nprand.normal(0, sigmas[1])
        c = old_pop[i][2] + nprand.normal(0, sigmas[2])

        new_population.append(np.array([a, b, c, *sigmas, 0]))
    return np.array(new_population)