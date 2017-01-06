
import numpy as np
from sklearn.datasets import fetch_mldata

NUM_TRAIN = 60000
NUM_DEBUG = 512

def load_mnist(debug=False):
    mnist = fetch_mldata('MNIST original')
    data = {}
    data['train_x'] = mnist['data'][:NUM_TRAIN]
    data['test_x'] = mnist['data'][NUM_TRAIN:]
    data['train_y'] = mnist['target'][:NUM_TRAIN]
    data['test_y'] = mnist['target'][NUM_TRAIN:]

    if debug:
        idxs = np.random.permutation(len(data['train_x']))
        data['train_x'] = data['train_x'][idxs][:NUM_DEBUG]
        data['train_y'] = data['train_y'][idxs][:NUM_DEBUG]

    return data

if __name__ == '__main__':
    load_mnist()