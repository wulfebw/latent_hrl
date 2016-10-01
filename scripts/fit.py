
import numpy as np

import generate_data
import hmm

def fit_hmm(data):
    k = 2
    max_iterations = 100
    threshold = 1e-5
    model = hmm.HMM(data, k, max_iterations, threshold, 
        verbose=True, seed=np.random.randint(100))
    model.initialize()
    log_prob, bic = model.fit()
    print model.A
    print model.B

if __name__ == '__main__':
    data = generate_data.generate_data()
    fit_hmm(data)
