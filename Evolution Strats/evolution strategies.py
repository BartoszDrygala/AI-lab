import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.random as nprand
import math

for lol in range(1):
    USE_FIXED_SEED = False

    if USE_FIXED_SEED:
        seed = 7585
    else:
        seed = random.randint(0, 10_000)

    random.seed(seed)
    np.random.seed(seed)

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
        initial_population[i, 3:6] = np.random.uniform(0, 10, 3)

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
        r1 = tau1 * nprand.normal(0, 1)
        for i in range(size):

            r2 = tau2 * nprand.normal(0, 1, 3)

            # Adaptive step size — allows exploration
            sigmas = np.maximum(
                old_pop[i][3:6] * np.exp(tau1 * r1 + tau2 * r2),
                1e-3  # prevents vanishing sigmas
            )

            sigmas = old_pop[i][3:6] * np.exp(tau1 * r1 + tau2 * r2)
            sigmas = np.clip(sigmas, 1e-3, 1)


            #if i < 3:  # print first 3 individuals only
                #print(f"sigmas[{i}] = {sigmas}")

            a = old_pop[i][0] + nprand.normal(0, sigmas[0])
            b = old_pop[i][1] + nprand.normal(0, sigmas[1])
            c = old_pop[i][2] + nprand.normal(0, sigmas[2])

            new_population.append(np.array([a, b, c, *sigmas, 0]))
        return np.array(new_population)

    # --- Evolution loop ---
    current_population = initial_population.copy()
    errors = err_vec(current_population)
    prev_best = np.min(errors)





    def is_good_enough(parent, offspring, eps=1e-5):

        result = np.abs(parent - offspring)
        #print(parent, offspring)
        return np.all(result < eps)

    generation = 0
    good_enough = False

    while not good_enough and generation < 200:
        generation += 1
        #print(f"Gen {generation}", flush=True)  #: good_enough={good_enough}")

        errors = err_vec(current_population)
        probs = make_Probs(errors)
        pool = make_Pool(current_population, probs, mu)
        offspring = make_New_Population(pool, mu)
        offspring_errors = err_vec(offspring)

        best_parent_idx = np.argmin(errors)
        best_offspring_idx = np.argmin(offspring_errors)
        best_parent = current_population[best_parent_idx, :3]
        best_offspring = offspring[best_offspring_idx, :3]

        good_enough = is_good_enough(best_parent, best_offspring, eps=1e-3)

        current_population = offspring

    print(f"Number: {lol}")
    print(f"Using random seed: {seed}")
    print(f"Generation number: {generation}")
    print(f"Diff: {np.abs(best_parent - best_offspring)}")

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

