
import csv
import matplotlib.pyplot as plt
import numpy as np

def load_data(input_filepath):
    data = []
    with open(input_filepath, 'rb') as infile:
        # remove header
        infile.readline()
        # read each line 
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            data.append(int(row[-1]))
    data = np.asarray(data, dtype=np.int64)
    return data

def log_sum_exp(values):
    max_value = max(values)
    if np.isinf(max_value):
        return -np.inf

    total = 0
    for v in values:
        total += np.exp(v - max_value)

    return np.log(total) + max_value

def plot_data(data):
    plt.plot(range(len(data)), data)
    plt.show()

if __name__ == '__main__':
    input_filepath = '../data/old_faithful.csv'
    data = load_data(input_filepath)
    plot_data(data)
    print data