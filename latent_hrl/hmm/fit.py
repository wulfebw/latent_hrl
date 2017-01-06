
import numpy as np

import data_utils
import generate_data
import hmm
import utils

def fit_poisson_hmm(data):
    k = 2
    max_iterations = 100
    threshold = 1e-5
    seed = 10 # np.random.randint(100)
    model = hmm.PoissonHMM(data, k, max_iterations, threshold, 
        verbose=True, seed=seed)
    log_prob, bic = model.fit()
    print model.A
    print model.B

def fit_multinomial_hmm(data):
    k = 3
    max_iterations = 100
    threshold = 1e-5
    seed = np.random.randint(100)
    model = hmm.MultinomialHMM(data, k, max_iterations, threshold, 
        verbose=True, seed=seed)
    log_prob, bic = model.fit()
    print 'A: ', model.A
    print 'B: ', model.B
    print 'log prob: {}\tbic: {}'.format(log_prob, bic)

if __name__ == '__main__':
    # data = generate_data.generate_poisson_data()
    # fit_poisson_hmm(data)
    # data = generate_data.generate_multinomial_data()

    # input_filepath = '/Users/wulfebw/Desktop/agent_0_actions.txt'
    # data = data_utils.load_action_data(input_filepath)
    data = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,0,0,0,
        1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0])
    fit_multinomial_hmm(data)
