import pickle
import warnings
import numpy as np
import time
from os import makedirs
from os.path import join


def save_result(results, data_name, folder):
    makedirs(folder, exist_ok=True)

    with open(join(folder, data_name) + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def number_of_points_in_interval(data, p_i, o_i):
    a, b = min(p_i, o_i), max(p_i, o_i)
    data_b = data[data < b]
    data_a_b = data_b[a < data_b]
    return len(data_a_b)


def missed_points_error(data, results, opt):
    return np.sum([number_of_points_in_interval(data, p_i, o_i) for p_i, o_i in zip(results, opt)])


def compute_quantiles_per_algorithm(data_function, quantiles, bounds, epsilon, algos):
    accuracy = np.zeros(len(algos))
    times = np.zeros(len(algos))
    kwargs = {'quantiles': quantiles, 'bounds': bounds, 'epsilon': epsilon}
    data = np.sort(data_function())
    kwargs.update({'array': data})
    opt = np.quantile(data, quantiles)

    for i, algo in enumerate(algos):
        t0 = time.time()
        ans = algo(**kwargs)
        t1 = time.time()
        times[i] = t1 - t0
        accuracy[i] = missed_points_error(data, ans, opt)

    return accuracy, times


def run_experiment(quantiles_function, data_function, algos, bounds=(0, 1), max_quantiles_size=120, epsilon=1):
    x = []
    accuracy = []
    times = []
    for qn in range(2, max_quantiles_size + 2):
        quantiles = quantiles_function(qn)
        a, t = compute_quantiles_per_algorithm(data_function, quantiles, bounds, epsilon, algos)
        accuracy.append(a)
        times.append(t)
        x.append(qn - 1)

    return np.array(x), np.array(accuracy).T, np.array(times).T


def run_avg_experiments(experiments, data_function, quantiles, bounds=(-100, 100), n_exp=100):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    all_accuracy = []
    all_times = []

    algos = [algo for _, algo in experiments]
    names = [name for name, _ in experiments]

    for i in range(n_exp):
        x, accuracy, times = run_experiment(quantiles, data_function, algos, bounds=bounds)
        all_accuracy.append(accuracy)
        all_times.append(times)

    all_accuracy = np.array(all_accuracy)
    all_accuracy = [(name, all_accuracy[:, i]) for i, name in enumerate(names)]
    all_times = np.array(all_times)
    all_times = [(name, all_times[:, i]) for i, name in enumerate(names)]

    return x, all_accuracy, all_times


def run_and_save(data, algos, output_folder, sub_sample_size):
    data, data_name = data()

    data_func = lambda: np.random.choice(data, sub_sample_size)
    quantiles_uniform = lambda size: np.sort(np.array([j / size for j in range(1, size)]))

    res = run_avg_experiments(algos, data_func, quantiles_uniform)
    save_result(res, data_name, output_folder)