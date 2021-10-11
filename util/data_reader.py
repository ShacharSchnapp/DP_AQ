import numpy as np


def uniform_data_reader(data_size=10000):
    data_name = 'Uniform'
    data_uniform = np.random.uniform(-5, 5, data_size)
    return data_uniform, data_name


def gaussian_data_reader(data_size=10000):
    data_name = 'Gaussian'
    gaussian_data = np.random.normal(0, 5, data_size)
    return gaussian_data, data_name
