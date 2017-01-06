
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, 'latent_hrl')
sys.path.append(os.path.abspath(path))

import hmm.data_utils
import hmm.generate_data as generate_data
import hmm.hmm


class TestHMM(unittest.TestCase):

    def test_forward(self):
        np.random.seed(3)

        data = np.array([1,10])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 10])
        m.forward()
        # check that the second class is more likely at the end
        self.assertTrue(m.alphas[-1,0] < m.alphas[-1,1])

        # multiple data case
        data = np.array([1,6,7,1])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 5])
        for tidx in range(m.T):
            for i in range(m.k):
                m.log_densities[tidx, i] = m.log_pdf(m.data[tidx], m.B[i])
        m.forward()
        # check that the first class is more likely at the end
        self.assertTrue(m.alphas[-1,0] > m.alphas[-1,1])

    def test_backward(self):
        data = np.array([10, 1, 1, 1])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[1,0],[0,1]])
        m.B = np.array([1, 10])
        for tidx in range(m.T):
            for i in range(m.k):
                m.log_densities[tidx, i] = m.log_pdf(m.data[tidx], m.B[i])
        m.backward()
        self.assertTrue(m.betas[0,0] < m.betas[0,1])

        data = np.array([1,1,1,1,5,5,5,5])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[1,0],[0,1]])
        m.B = np.array([1, 5])
        for tidx in range(m.T):
            for i in range(m.k):
                m.log_densities[tidx, i] = m.log_pdf(m.data[tidx], m.B[i])
        m.backward()
        self.assertTrue(m.betas[0, 0] < m.betas[0, 1])

    def test_e_step(self):
        data = np.array([1,10,1,10,1,10])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 10])
        m.e_step()
        print m.gammas
        print m.etas
        # what even is gamma?

    def test_e_step_comparison(self):
        data = np.array([1,10,1,10,1,10])
        k = 2
        max_iterations = 10
        threshold = 1e-5

        np.random.seed(2)
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 10])
        m.e_step()
        g1 = m.gammas
        e1 = m.etas

        np.random.seed(2)
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 10])
        m.e_step()
        g2 = m.gammas
        e2 = m.etas

        print g1 - g2
        print e1 - e2

    def test_m_step(self):
        data = np.array([1,10,1,10,1,10,1,10])
        # data = np.array([1, 1])
        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.initialize()
        m.pi = np.array([.5,.5])
        m.A = np.array([[.5,.5],[.5,.5]])
        m.B = np.array([1, 10])
        for _ in range(10):
            m.e_step()
            m.m_step()
        print m.A
        print m.B
        print m.pi
        # what even is gamma?

    def test_hmm_on_generated_data(self):
        np.random.seed(2)

        # k = 2 case
        A = np.array([[0,1],[1,0]])
        B = np.array([[2],[10]])
        pi = np.array([.5,.5])
        T = 100
        dist = np.random.poisson
        data = generate_data.generate_data(A, B, pi, T, dist)
        print data

        k = 2
        max_iterations = 10
        threshold = 1e-5
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.fit()
        print 'actual A: {}'.format(A)
        print 'learned A: {}'.format(m.A)
        print 'actual B: {}'.format(B)
        print 'learned B: {}'.format(m.B)

        # k = 3 case
        A = np.array([[.33,.33,.33],[.33,.33,.33],[.33,.33,.33]])
        B = np.array([[2],[10],[20]])
        pi = np.array([.5,.5,.5])
        T = 100
        dist = np.random.poisson
        data = generate_data.generate_data(A, B, pi, T, dist)
        print data

        k = 3
        max_iterations = 50
        threshold = 1e-5
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold)
        m.fit()
        print 'actual A: {}'.format(A)
        print 'learned A: {}'.format(m.A)
        print 'actual B: {}'.format(B)
        print 'learned B: {}'.format(m.B)

        # k = 4 case
        A = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
        B = np.array([[2],[10],[20],[30]])
        pi = np.array([.25,.25,.25,.25])
        T = 50
        dist = np.random.poisson
        data = generate_data.generate_data(A, B, pi, T, dist)
        print data

        k = 4
        max_iterations = 50
        threshold = 1e-10
        m = hmm.hmm.PoissonHMM(data, k, max_iterations, threshold, verbose=True, seed=np.random.randint(100))
        m.fit()
        print 'actual A: {}'.format(A)
        print 'learned A: {}'.format(m.A)
        print 'actual B: {}'.format(B)
        print 'learned B: {}'.format(m.B)

# class TestHMMRealData(unittest.TestCase):

#     def get_data(self):
#         input_filepath = '../data/old_faithful.csv'
#         data = utils.load_data(input_filepath)
#         return data

#     def test_forward(self):
#         data = self.get_data()
#         k = 2
#         max_iterations = 10
#         threshold = 1e-5
#         m = hmm.HMM(data, k, max_iterations, threshold)
#         m.initialize()
#         m._forward()
#         print m.alphas

#     def test_hmm_on_real_data(self):
#         np.random.seed(42)

#         data = self.get_data()

#         num_k = 9
#         num_runs = 5
#         max_iterations = 50
#         threshold = 1e-4
#         output_filepath = '../data/hmm_results.npz'

#         bics, icls, log_probs = [], [], []
#         As, Bs = [], []

#         for k in range(2, num_k + 1):
#             best_A, best_B = None, None
#             best_log_prob, best_bic = -sys.maxint, -sys.maxint
#             for r in range(num_runs):

#                 model = hmm.HMM(data, k, max_iterations, threshold, verbose=True, seed=np.random.randint(100))
#                 model.initialize()
#                 log_prob, bic = model.fit()
        
#                 if bic > best_bic:
#                     best_bic = bic
#                     best_log_prob = log_prob
#                     best_A = copy.deepcopy(model.A)
#                     best_B = copy.deepcopy(model.B)

#             As.append(best_A)
#             Bs.append(best_B)
#             bics.append(best_bic)
#             log_probs.append(best_log_prob)
#             np.savez(output_filepath, As=As, Bs=Bs, bics=bics, log_probs=log_probs)
        
#             plt.plot(range(len(log_probs)), log_probs, label='log_probs')
#             plt.title('log_probs')
#             plt.savefig('../data/hmm_log_probs.png')
#             plt.close()

#             plt.plot(range(len(bics)), bics, label='bics')
#             plt.title('bics')
#             plt.savefig('../data/hmm_bics.png')
#             plt.close()

# def analyze_save():
#     d = np.load('../data/hmm_results.npz')
#     log_probs = d['log_probs']
#     bics = d['bics']
#     As = d['As']
#     Bs = d['Bs']

#     plt.plot(range(2, len(log_probs) + 2), log_probs, label='log_probs', linestyle='--')
#     plt.plot(range(2, len(bics) + 2), bics, label='bics', linestyle='-')
#     plt.title('hmm model selection')
#     plt.legend(loc='best')
#     plt.savefig('../data/hmm_model_selection.png')
#     plt.close()

#     best_idx = np.argmax(bics)
#     print 'best k: {}'.format(best_idx + 2)
#     print 'A: {}'.format(As[best_idx])
#     print 'B: {}'.format(Bs[best_idx])

if __name__ == '__main__':
    # analyze_save()
    unittest.main()