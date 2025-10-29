import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.random as nprand
import math

# --- Reproducibility ---
random.seed(4)
np.random.seed(4)

# --- Load data ---
df = pd.read_fwf('ES_data_7.dat', names=['x', 'y'])
x, y = df['x'].to_numpy(), df['y'].to_numpy()

# --- Parameters ---
mu = 150
n = 3
tau1 = 1 / math.sqrt(2 * n)
tau2 = 1 / math.sqrt(2 * math.sqrt(n))

#Initial population: [a, b, c, σa, σb, σc, dummy]
initial_population = np.zeros((mu, 7))
for i in range(mu):
    initial_population[i, :3] = np.random.uniform(-10, 10, 3)
    initial_population[i, 3:6] = np.random.uniform(0.1, 2, 3)

# function from the task
def fun(a, b, c, i):
    return a * (i**2 - b * math.cos(c * math.pi * i))


def err_vec(vectors):
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
def make_New_Population(old_pop, size):
    new_population = []
    for i in range(size):
        r1 = nprand.normal(0, 1)
        r2 = nprand.normal(0, 1, 3)

        # Adaptive step size — allows exploration
        sigmas = np.maximum(
            old_pop[i][3:6] * np.exp(tau1 * r1 + tau2 * r2),
            1e-3  # prevents vanishing sigmas
        )

        sigmas = old_pop[i][3:6] * np.exp(tau1 * r1 + tau2 * r2)
        a = old_pop[i][0] + nprand.normal(0, sigmas[0])
        b = old_pop[i][1] + nprand.normal(0, sigmas[1])
        c = old_pop[i][2] + nprand.normal(0, sigmas[2])

        new_population.append(np.array([a, b, c, *sigmas, 0]))
    return np.array(new_population)

# --- Evolution loop ---
current_population = initial_population.copy()
errors = err_vec(current_population)
prev_best = np.min(errors)
diff = float('inf')
generation = 0
best_error = 1

while (diff > 1e-5 or generation < 20)  and generation < 200:
    errors = err_vec(current_population)
    probs = make_Probs(errors)
    pool = make_Pool(current_population, probs, mu)
    offspring = make_New_Population(pool, mu)

    combined = np.vstack((current_population, offspring))
    combined_errors = err_vec(combined)
    best_idx = np.argsort(combined_errors)[:mu]

    current_population = combined[best_idx]
    best_error = combined_errors[best_idx[0]]

    diff = abs(prev_best - best_error) / (abs(prev_best) + 1e-8)
    prev_best = best_error
    generation += 1

    print(f"Gen {generation}: best_error={best_error:.6f}, diff={diff:.6f}")

print(f"\nStopped after {generation} generations, best error = {best_error:.6f}")

# --- Plot best result ---
best = current_population[0]
xs = np.linspace(-5, 5, 150)
ys = [fun(best[0], best[1], best[2], xi) for xi in xs]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=10, label="True data")
plt.scatter(xs, ys, s=10, c='red', label="Best function")
plt.xlabel('i')
plt.ylabel('o(i)')
plt.title('True values vs function')
plt.grid(True)
plt.legend()
plt.show()