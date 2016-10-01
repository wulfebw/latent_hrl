
import numpy as np

import utils

def poisson_density(point, mean):
    return mean ** point * np.exp(-mean) / np.math.factorial(point)

def log_factorial(value):
    return np.sum(np.log(v) for v in range(1, int(value) + 1, 1))

def log_poisson_density(point, mean):
    if mean <= 0:
        raise ValueError('mean value must be > 0, got : {}'.format(mean))
    return point * np.log(mean) - mean - log_factorial(point)

class HMM(object):

    def __init__(self, data, k, max_iterations, threshold, verbose=True, seed=1):
        self.data = data
        self.k = k
        self.T = data.shape[0]
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.verbose = verbose
        np.random.seed(seed)

    def initialize(self):
        # unpack dimensions
        T, k = self.T, self.k

        # initialize model parameters
        unnormalized_pi = np.random.rand(k)
        self.pi = unnormalized_pi / np.sum(unnormalized_pi)
        unnormalized_A = np.random.rand(k, k) 
        self.A = unnormalized_A / np.sum(unnormalized_A, axis=1, keepdims=True)
        self.B = np.float64(np.random.randint(low=0, high=np.max(self.data), size=(k)))

        # allocate responsibilities containers
        self.alphas = np.empty((T, k))
        self.betas = np.empty((T, k))

        # wiki
        self.gammas = np.empty((T, k))
        self.etas = np.empty((T, k, k))

    def forward(self):
        """
        The forward pass computes for each sample, for each timestep, 
        and for each latent class, the probability that the latent state
        was the latent class, and stores these values in self.alphas.
        
        This is accomplished using a dynamic programming approach that takes
        advantage of the assumption that the future depends only upon the previous 
        timestep. Specifically, it iterates through each sequence keeping track
        of the probability of each class up until that timestep. Then, to compute
        the probability of each time step at t + 1, it sums over a set of 
        probabilities where each is the probability of transitioning from a 
        previous class times the probability of the current class given the 
        observation times the probability of the previous class. This sum gives
        the total probability of being in a certain class at timestep t + 1.
        """
        # initialize first timestep value of alpha for each sample 
        # to the start probability of the corresponding latent class in A
        self.alphas[0, :] = np.log(self.pi) + self.log_densities[0, :]

        # allocate a buffer to reuse in inner loop
        timestep_values = np.empty(self.k)

        # tidx starts at 1 since zeroth timestep 
        # of alphas has already been initialized
        for tidx, value in enumerate(self.data[1:], 1):

            # iterate over k values to fill
            for j in range(self.k):

                # iterate over previous k values
                for i in range(self.k):
                    timestep_values[i] = self.log_A[i, j] + self.alphas[tidx - 1, i]

                # set value for jth class at time t
                self.alphas[tidx, j] = utils.log_sum_exp(timestep_values) + self.log_densities[tidx, j]

    def backward(self):
        # initialize first timestep value of beta to zero
        # and then iterate backward starting from the end
        # zero because operating in log space
        self.betas[-1, :] = 0

        # allocate a buffer to reuse in inner loop
        timestep_values = np.empty(self.k)

        # start from second to last
        for tidx in range(self.T - 2, -1, -1):

            # iterate over k values to fill (timestep t)
            # note that i and j are flipped from forward pass
            for i in range(self.k):

                # iterate over next k values (timestep t + 1)
                for j in range(self.k):
                    timestep_values[j] = self.log_densities[tidx + 1, j] + self.log_A[i, j] + self.betas[tidx + 1, j]

                # set value for jth class at time t
                self.betas[tidx, i] = utils.log_sum_exp(timestep_values)

    def e_step(self):
        # precompute information
        self.log_A = np.log(self.A)
        self.log_densities = np.empty((self.T, self.k))
        for tidx in range(self.T):
            for i in range(self.k):
                self.log_densities[tidx, i] = log_poisson_density(self.data[tidx], self.B[i])

        # compute alphas and betas
        self.forward()
        self.backward()

        # gammas: probability of being in state i at time t
        self.gammas = self.alphas + self.betas
        # normalize
        for tidx in range(self.T):
            self.gammas[tidx, :] -= utils.log_sum_exp(self.gammas[tidx, :])
        # convert from log to normal space
        self.gammas = np.exp(self.gammas)

        # etas: probability of being in state i and j at times t and t+1
        for tidx in range(self.T - 1):
            for i in range(self.k):
                for j in range(self.k):
                    self.etas[tidx, i, j] = self.alphas[tidx, i] + self.log_A[i, j] + self.log_densities[tidx + 1, j] + self.betas[tidx + 1, j]
        # normalize
        self.etas -= utils.log_sum_exp(self.alphas[-1, :])
        # convert from log to normal space
        self.etas = np.exp(self.etas)

        # the log sum exp of the last alphas gives the log prob of the full sequence
        return utils.log_sum_exp(self.alphas[-1, :])

    def m_step(self):
        # pi
        self.pi = self.gammas[0, :]

        # transition probabilities
        for i in range(self.k):
            for j in range(self.k):
                self.A[i, j] = np.sum(self.etas[:-1, i, j]) / np.sum(self.gammas[:-1, i])

        # emission probabilities
        self.B = np.sum(self.data[:, np.newaxis] * self.gammas, axis=0) 
        self.B /= np.sum(self.gammas, axis=0)

    def fit(self):

        # initialize parameter estimates
        self.initialize()

        # run e_step, m_step for max iterations or until convergence
        prev_log_prob = log_prob = 0
        for idx in range(self.max_iterations):

            # e-step
            log_prob = self.e_step()

            # m-step
            self.m_step()

            # check for convergence
            if abs(log_prob - prev_log_prob) < self.threshold: 
                break
            prev_log_prob = log_prob
            if self.verbose: 
                print 'iter: {}\tlog_prob: {:.4f}'.format(idx, log_prob)

        # return the log probability of the fit

        num_params = self.k - 1 + self.k * (self.k - 1)
        num_samples = len(self.data)
        bic = log_prob - num_params / 2. * np.log(num_samples) 
        return log_prob, bic

