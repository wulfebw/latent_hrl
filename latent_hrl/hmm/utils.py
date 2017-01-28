
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

def log_multinomial_density(point, probs):
    return np.log(probs[point])

def compute_multinomial_classes(data):
    """
    Description:
        - given a list of discrete values taking on values 
            between 0 and m - 1, return m.

    Args:
        - data: a row vector of discrete values

    Returns:
        - m: the number of classes
    """
    if type(data[0]) != int:
        temp_data = np.hstack(data)
        assert min(temp_data) == 0, "discrete classes must start at 0"
        return max(temp_data) + 1
    else:
        assert min(data) == 0, "discrete classes must start at 0"
        assert np.array_equal([int(v) for v in data], data)
        return max(data) + 1
