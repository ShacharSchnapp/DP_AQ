import numpy as np

from algorithms.approximate_quantiles_algo import gaussian_noise
from google_research.dp_multiq.ind_exp import ind_exp, opt_comp_calculator
from google_research.dp_multiq.joint_exp import joint_exp
from google_research.dp_multiq.tree import tree


def run_agg_tree(array, quantiles, bounds, epsilon, swap=False, delta=1e-6):
    array, bounds = gaussian_noise(array, bounds)
    return tree(array, qs=quantiles, data_low=bounds[0], data_high=bounds[1], eps=epsilon, delta=delta, swap=swap)


def run_appind_exp(array, quantiles, bounds, epsilon, swap=False, delta=1e-6):
    array, bounds = gaussian_noise(array, bounds)
    m = len(quantiles)
    return ind_exp(array, qs=quantiles, data_low=bounds[0], data_high=bounds[1],
                   divided_eps=opt_comp_calculator(epsilon, delta, m), swap=swap)


def run_joint_exp(array, quantiles, bounds, epsilon, swap=False):
    array, bounds = gaussian_noise(array, bounds)
    return joint_exp(np.sort(array), qs=quantiles, data_low=bounds[0], data_high=bounds[1], eps=epsilon, swap=swap)
