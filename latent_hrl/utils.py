
import numpy as np

def log_sum_exp(values):
    max_value = max(values)
    if np.isinf(max_value):
        return -np.inf

    total = 0
    for v in values:
        total += np.exp(v - max_value)

    return np.log(total) + max_value

def poisson_density(point, mean):
    return mean ** point * np.exp(-mean) / np.math.factorial(point)

def log_factorial(value):
    return np.sum(np.log(v) for v in range(1, int(value) + 1, 1))

def log_poisson_density(point, mean):
    if mean <= 0:
        raise ValueError('mean value must be > 0, got : {}'.format(mean))
    return point * np.log(mean) - mean - log_factorial(point)