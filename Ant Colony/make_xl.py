import pandas as pd
import numpy as np

def make_xl(filename, x, y):
    dist_matrix = np.zeros((len(x), len(x)))
    for i in range(0, len(x)):
        for j in range(0, len(x)):
            dist_matrix[i][j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

    df = pd.DataFrame(np.array(dist_matrix))

    df.to_excel(f"{filename}")
