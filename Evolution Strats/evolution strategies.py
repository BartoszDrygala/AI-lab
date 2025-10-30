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

# function from the task


# --- Evolution loop ---
current_population = initial_population.copy()
errors = err_vec(current_population, x,y)
prev_best = np.min(errors)
diff = float('inf')
generation = 0
best_error = 1
pros = []
while abs(diff) > 0.00005 and generation < 200:
    errors = err_vec(current_population,x,y)
    probs_parent = make_Probs(errors)
    best_idx_parent = np.argsort(errors)[0]
    best_parent = current_population[best_idx_parent][:3]
    print('biggest probability',probs_parent.max())
    pool = make_Pool(current_population, probs_parent, mu)
    current_population = make_New_Population(pool, mu, tau1, tau2)
    errors_children = err_vec(current_population, x, y)
    probs_child = make_Probs(errors_children)
    best_idx_child = np.argsort(errors_children)[0]
    best_child = current_population[best_idx_child][:3]
    diff = np.linalg.norm(best_child-best_parent)
    best_error = current_population[best_idx_child][6]
    '''
    combined = np.vstack((current_population, offspring))
    combined_errors = err_vec(combined)
    probs_combined = make_Probs(combined_errors)

    
    print('best combined: ', best_combined)
    best_idx = np.argsort(combined_errors)[:mu]

    current_population = combined[best_idx]
    best_combined = 

    diff = best_combined - best_parent'''
    generation += 1

    print(f"Gen {generation}: best error={best_error}, diff={diff:.6f}")

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