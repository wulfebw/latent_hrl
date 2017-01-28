
import numpy as np

import utils

class MultisampleHMM(object):

    def __init__(self, data, k, max_iterations, threshold=1e-6, verbose=True, seed=1):
        self.data = data
        self.k = k
        self.num_samples = len(data)
        self.Ts = [sample.shape[0] for sample in data]
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.verbose = verbose
        np.random.seed(seed)

    def forward(self, sample_idx):
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
        self.alphas[sample_idx][0, :] = np.log(self.pi) + self.log_densities[0, :]

        # allocate a buffer to reuse in inner loop
        timestep_values = np.empty(self.k)

        # tidx starts at 1 since zeroth timestep 
        # of alphas has already been initialized
        for tidx, value in enumerate(self.data[sample_idx][1:], 1):

            # iterate over k values to fill
            for j in range(self.k):

                # iterate over previous k values
                for i in range(self.k):
                    timestep_values[i] = self.log_A[i, j] + self.alphas[sample_idx][tidx - 1, i]

                # set value for jth class at time t
                self.alphas[sample_idx][tidx, j] = utils.log_sum_exp(timestep_values) + self.log_densities[tidx, j]

    def backward(self, sample_idx):
        # initialize first timestep value of beta to zero
        # and then iterate backward starting from the end
        # zero because operating in log space
        self.betas[sample_idx][-1, :] = 0

        # allocate a buffer to reuse in inner loop
        timestep_values = np.empty(self.k)

        # start from second to last
        for tidx in range(self.Ts[sample_idx] - 2, -1, -1):

            # iterate over k values to fill (timestep t)
            # note that i and j are flipped from forward pass
            for i in range(self.k):

                # iterate over next k values (timestep t + 1)
                for j in range(self.k):
                    timestep_values[j] = self.log_densities[tidx + 1, j] + self.log_A[i, j] + self.betas[sample_idx][tidx + 1, j]

                # set value for jth class at time t
                self.betas[sample_idx][tidx, i] = utils.log_sum_exp(timestep_values)

    def e_step(self, sample_idx):
        # unpack info for this sample
        T = self.Ts[sample_idx]
        data = self.data[sample_idx]

        # precompute information
        self.log_A = np.log(self.A)
        self.log_densities = np.zeros((T, self.k))
        for tidx in range(T):
            for i in range(self.k):
                self.log_densities[tidx, i] = self.log_pdf(data[tidx], self.B[i])

        # compute alphas and betas
        self.forward(sample_idx)
        self.backward(sample_idx)

        # gammas: probability of being in state i at time t
        self.gammas[sample_idx] = self.alphas[sample_idx] + self.betas[sample_idx]
        # normalize
        for tidx in range(T):
            self.gammas[sample_idx][tidx, :] -= utils.log_sum_exp(self.gammas[sample_idx][tidx, :])
        # convert from log to normal space
        self.gammas[sample_idx] = np.exp(self.gammas[sample_idx])

        # etas: probability of being in state i and j at times t and t+1
        for tidx in range(T - 1):
            for i in range(self.k):
                for j in range(self.k):
                    self.etas[sample_idx][tidx, i, j] = self.alphas[sample_idx][tidx, i] + self.log_A[i, j] + self.log_densities[tidx + 1, j] + self.betas[sample_idx][tidx + 1, j]
        # normalize
        self.etas[sample_idx] -= utils.log_sum_exp(self.alphas[sample_idx][-1, :])
        # convert from log to normal space
        self.etas[sample_idx] = np.exp(self.etas[sample_idx])

        # the log sum exp of the last alphas gives the log prob of the full sequence
        return utils.log_sum_exp(self.alphas[sample_idx][-1, :])

    def fit(self):

        # initialize parameter estimates
        self.initialize()

        # run e_step, m_step for max iterations or until convergence
        prev_log_prob = log_prob = 0
        for idx in range(self.max_iterations):

            # e-step
            log_prob = 0
            for i in range(self.num_samples):
                log_prob += self.e_step(i)

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

class MultisampleMultinomialHMM(MultisampleHMM):

    def __init__(self, data, k, max_iterations, threshold=1e-6, 
            verbose=True, seed=1):
        super(MultisampleMultinomialHMM, self).__init__(
            data, k, max_iterations, threshold, verbose, seed)
        self.log_pdf = utils.log_multinomial_density

        # in multinomial case, we assume that the output is discrete and 
        # that it can take on 1 of m values. We extract m from the data
        self.m = utils.compute_multinomial_classes(data)

    def initialize(self):
        # unpack dimensions
        N, Ts, k, m = self.num_samples, self.Ts, self.k, self.m

        # initialize model parameters
        unnormalized_pi = np.random.rand(k)
        self.pi = unnormalized_pi / np.sum(unnormalized_pi)

        # transition probabilities
        unnormalized_A = np.random.rand(k, k) 
        self.A = unnormalized_A / np.sum(unnormalized_A, axis=1, keepdims=True)
        
        # emission probabilities for m discrete outputs for k latent classes
        unnormalized_B = np.random.rand(k, m)
        self.B = np.float64(
            unnormalized_B / np.sum(unnormalized_B, axis=1, keepdims=True))

        # allocate responsibilities containers
        self.alphas = [np.empty((T, k)) for T in Ts]
        self.betas = [np.empty((T, k)) for T in Ts]

        # wiki
        self.gammas = [np.empty((T, k)) for T in Ts]
        self.etas = [np.empty((T, k, k)) for T in Ts]

    def m_step(self):
        # pi
        self.pi = np.mean([g[0, :] for g in self.gammas])

        # transition probabilities
        for gammas, etas in zip(self.gammas, self.etas):
            for i in range(self.k):
                for j in range(self.k):
                    self.A[i, j] = np.sum(etas[:-1, i, j]) / np.sum(gammas[:-1, i])

        # emission probabilities
        # B shape is (k, m)
        # gammas shape is (Ts, T, k)
        # to get new B, I want to focus on the specific latent class (row of B)
        # then sum over time, where I add one to the latent class row in the position 
        # of the observed variable, multiplying by gamma of this timestep and this class
        # then normalize at the end
        new_B = np.zeros((self.k, self.m))
        for class_idx in range(self.k):
            for sidx in range(self.num_samples):
                for tidx in range(self.Ts[sidx]):
                    new_B[class_idx, self.data[sidx][tidx]] += self.gammas[sidx][tidx, class_idx]

                    print('class_idx: {}'.format(class_idx))
                    print('sidx: {}'.format(sidx))
                    print('tidx: {}'.format(tidx))
                    print('self.data[sidx][tidx]: {}'.format(self.data[sidx][tidx]))
                    print('self.gammas[sidx][tidx, class_idx]: {}'.format(self.gammas[sidx][tidx, class_idx]))
                    print('new_B: {}'.format(new_B))
                    raw_input()
            new_B[class_idx, :] /= sum(
                [sum(g[:, class_idx]) for g in self.gammas])
        self.B = new_B
            