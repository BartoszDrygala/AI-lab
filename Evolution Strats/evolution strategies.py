import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.random as nprand
import math
from methods import err_vec,make_Probs, make_Pool, make_New_Population, fun

# --- Reproducibility ---
#random.seed(4)
#np.random.seed(4)

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



# --- Evolution loop ---
current_population = initial_population.copy()
errors = err_vec(current_population, x,y)
prev_best = np.min(errors)
diff = float('inf')
generation = 0
best_error = 1
pros = []
best_parent = []


# main loop, change to difference between best and best new
while abs(diff) > 10e-5 and generation < 200:

    # evalue erros/fittnes of current population
    errors = err_vec(current_population,x,y)
    current_population[:,6] = errors
    # collect the highest fitness parent
    best_parent = current_population[np.argmin(errors)]
    # convert those error to probabilties for rulette wheel
    probs_parent = make_Probs(errors)
    #make a mating pool based on calcuated probabilites and creatre a new population
    pool = make_Pool(current_population, probs_parent, mu)
    offspring = make_New_Population(pool, mu, tau1, tau2)
    
    # combine both population into single big one
    combined_populations = np.vstack((current_population, offspring))
    # create their errors
    combined_errors = err_vec(combined_populations, x ,y)
    combined_populations[:,6] = combined_errors
    # sort new population by errors to select mu best
    new_population = combined_populations[np.argsort(combined_errors)][:mu]
    # get best individual and their error
    best_original_combined = new_population[0]
    
    #check the second best if first is the same
    if np.allclose(best_original_combined[:6], best_parent[:6], atol=1e-8):
        best_original_combined = new_population[1]

    best_error = best_original_combined[6]
    # calucalting differece between best fitness levels between both generation
    diff = 1/best_original_combined[6] - 1/best_parent[6]
    # finishing the loop
    current_population = new_population
    generation += 1



# evalue if succeded
print(f"\nStopped after {generation} generations, best error = {best_error:.6f}")
print(f'coefficients: {best_parent[0]} {best_parent[1]} {best_parent[2]}')

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