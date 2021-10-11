from algorithms.approximate_quantiles_algo import approximate_quantiles_algo
import numpy as np

data_size = 100000
n_quantiles = 3
quantiles_uniform = np.sort(np.array([i / (n_quantiles + 1) for i in range(1, n_quantiles + 1)]))
print('Optimal quantiles: ', quantiles_uniform)
dp_approx_quantiles = approximate_quantiles_algo(array=np.random.uniform(0, 1, data_size),
                                                 quantiles=quantiles_uniform,
                                                 bounds=[0, 1],
                                                 epsilon=1)
print('Differentially private quantiles approximation quantiles: ', dp_approx_quantiles)
