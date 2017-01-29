
import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir, 'latent_hrl')
sys.path.append(os.path.abspath(path))

import rl.algorithms as algorithms
# from hmm.hmm import MultinomialHMM
from hmmlearn import hmm
import utils

class ScratchAlgorithm(object):

    def __init__(self, actions, discount, explorationProb, stepSize,
            max_k=6, max_iterations=100, tol=1e-3, enlighten_threshold=1e-3,
            enlighten_steps=100, init_enlighten_steps=50000, max_levels=2):
        # hierarchy
        self.levels = [algorithms.QLearningAlgorithm(actions, discount, 
            explorationProb, stepSize)]
        self.num_levels = 1
        self.level_idx = 0
        self.latent_states = [0]
        self.max_k = max_k
        self.max_iterations = max_iterations
        self.currently_enlightening = False
        self.enlighten_threshold = enlighten_threshold
        self.enlighten_steps = enlighten_steps
        self.steps_since_prev_enlighten = 0
        self.tol = tol
        self.init_enlighten_steps = init_enlighten_steps
        self.max_levels = max_levels

        # learner hyperparams
        self.discount = discount
        self.explorationProb = explorationProb
        self.stepSize = stepSize

        # dataset
        self.actions = []
        self.episode_actions = []
        self.cur_return = 0
        self.last_action_state = None

    def step(self, x):
        # print('level_idx: {}\tlatent states: {}'.format(self.level_idx, self.latent_states))
        self.steps_since_prev_enlighten += 1
        x = tuple(x)
        if self.level_idx == self.num_levels - 1:
            # print('top level')
            self._incorporate_feedback(x)
            next_z = self.levels[self.level_idx].getAction(x)
            # print('high level action: {}'.format(next_z))
            self.last_action_state = x
            # print('setting last action state: {}'.format(self.last_action_state))
            self.actions.append([next_z])
            for i in range(self.num_levels - 2, -1, -1):
                self.latent_states[i] = next_z
                next_z = self.emission(i)
                # print('step down from top level')
                # print('i: {}\temission: {}'.format(i, next_z))
            action = next_z
            # print('exiting top level, action: {}'.format(action))
            self.level_idx = 0
        else:
            # print('{} level'.format(self.level_idx))
            next_z = self.transition(self.level_idx)
            if next_z != self.latent_states[self.level_idx]:
                self.level_idx += 1
                return self.step(x)
            else:
                next_z = self.emission(self.level_idx)
                # IS THIS NEEDED? I think not
                # self.actions.append([next_z])
                for i in range(self.level_idx - 1, -1, -1):
                    self.latent_states[i] = next_z
                    next_z = self.emission(i)
            action = next_z
            self.level_idx = 0

        # print('returning from step, action: {}'.format(action))
        return action

    def incorporate_feedback(self, r, t):
        if (self.currently_enlightening 
                and len(self.episode_actions) > self.enlighten_steps):
            self.currently_enlightening = False
            self.enlighten()
            self.actions = []
        else:
            self.cur_return += r
            if t:
                self.episode_actions.append(self.actions)
                self.level_idx = self.num_levels - 1
                self._incorporate_feedback(None)
                self.actions = []

    def _incorporate_feedback(self, nx):
        # print('self.last_action_state: {}'.format(self.last_action_state))
        # print('self.actions: {}'.format(self.actions))
        nx = tuple(nx) if nx is not None else None
        if self.last_action_state is not None and len(self.actions) >= 1:
            x = self.last_action_state
            a = self.actions[-1][0]
            r = self.cur_return
            loss = self.levels[-1].incorporate_feedback(x, a, r, nx)
            if (loss < self.enlighten_threshold 
                    and self.steps_since_prev_enlighten > self.init_enlighten_steps
                    and self.num_levels < self.max_levels):
                self.attain_enlightenment()
        self.cur_return = 0
        self.last_action_state = nx

    def attain_enlightenment(self):
        self.levels[-1].explorationProb = 0.01
        self.currently_enlightening = True
        self.actions = []
        self.episode_actions = []
        self.steps_since_prev_enlighten = 0

    def enlighten(self):
        print(self.episode_actions)
        data = np.concatenate(self.episode_actions)
        lengths = [len(x) for x in self.episode_actions]
        models = []
        best_model_idx, best_log_prob = 0, -sys.maxint
        for i, k in enumerate(range(2, self.max_k + 1)):
            m = hmm.MultinomialHMM(n_components=k, n_iter=self.max_iterations, 
                tol=self.tol)
            models.append(m)
            m.fit(data, lengths)
            log_prob = np.sum(np.sum(m._compute_log_likelihood(d)) for d in self.episode_actions)
            print('k: {}\tlikelihood: {}'.format(k, log_prob))
            if log_prob > best_log_prob:
                best_model_idx = i
                best_log_prob = log_prob
                best_k = k
        self.levels[-1] = models[best_model_idx]
        m = models[best_model_idx]
        print(m.transmat_)
        print(m.startprob_)
        print(m.emissionprob_)
        # raw_input()
        self.levels.append(algorithms.QLearningAlgorithm(range(best_k), self.discount, 
            self.explorationProb, self.stepSize))
        self.latent_states.append(0)
        self.num_levels += 1

        # reset dataset
        self.actions = []

    def emission(self, i):
        assert i >= 0 and i <= self.num_levels
        probs = self.levels[i].emissionprob_[self.latent_states[i], :]
        action = utils.sample(probs)
        # print('emission action: {}'.format(action))
        # raw_input()
        return action

    def transition(self, i):
        assert i >= 0 and i <= self.num_levels
        probs = self.levels[i].transmat_[self.latent_states[i], :]
        next_z = utils.sample(probs)
        # print(self.latent_states)
        # print(i)
        # print(self.latent_states[i])
        # print('transition next_z: {}'.format(next_z))
        # raw_input()
        return next_z


