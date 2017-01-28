
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'latent_hrl')
sys.path.append(os.path.abspath(path))

# import hmm.multisample_hmm
from hmmlearn import hmm

class TestHMM(unittest.TestCase):

    def test_fit(self):
        np.random.seed(2)

        # k = 2 case
        data = [np.array(a) for a in [[[1],[0],[1]],[[0],[1],[0]]]]
        k = 2
        # max_iterations = 100
        # threshold = 1e-8
        # m = hmm.multisample_hmm.MultisampleMultinomialHMM(
        #     data, k, max_iterations, threshold)
        # m.fit()
        # print 'learned A: {}'.format(m.A)
        # print 'learned B: {}'.format(m.B)
        X = np.concatenate(data)
        print(X)
        lengths = [len(d) for d in data]
        m = hmm.MultinomialHMM(n_components=k)
        m.fit(X, lengths)
        print(m.transmat_)
        print(m.startprob_)
        print(m.emissionprob_)
        print(np.sum(m._compute_log_likelihood(x) for x in data))


if __name__ == '__main__':
    unittest.main()