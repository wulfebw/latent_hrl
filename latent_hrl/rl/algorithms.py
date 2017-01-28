
import cPickle
import collections
import copy
import itertools
import numpy as np
import os
import random
import time

class RLAlgorithm(object):
    """
    :description: abstract class defining the interface of a RL algorithm
    """

    def getAction(self, state):
        raise NotImplementedError("Override me")

    def incorporateFeedback(self, state, action, reward, newState):
        raise NotImplementedError("Override me")

class ValueLearningAlgorithm(RLAlgorithm):
    """
    :description: base class for RL algorithms that approximate the value function.
    """
    def __init__(self, actions, discount, explorationProb, stepSize):
        """
        :type: actions: list
        :param actions: possible actions to take

        :type discount: float
        :param discount: the discount factor

        :type explorationProb: float
        :param explorationProb: probability of taking a random action

        :type stepSize: float
        :param stepSize: learning rate
        """
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 1
        self.stepSize = stepSize

    def feature_extractor(self, state, action):
        """
        :description: this is the identity feature extractor, so we use tables here for the function
        """
        return [((state, action), 1)]

    def getQ(self, state, action):
        """
        :description: returns the Q value associated with this state-action pair

        :type state: dictionary
        :param state: the state of the game

        :type action: int
        :param action: the action for which to retrieve the Q-value
        """
        score = 0
        for f, v in self.feature_extractor(state, action):
            score += self.weights[f] * v
        return score

    def getAction(self, state):
        """
        :description: returns an action accoridng to epsilon-greedy policy

        :type state: dictionary
        :param state: the state of the game
        """
        self.numIters += 1

        if random.random() < self.explorationProb:
            return random.choice(self.actions)
        else:
            maxAction = max((self.getQ(state, action), action) for action in self.actions)[1]
        return maxAction

    def getStepSize(self):
        """
        :description: return the step size
        """
        return self.stepSize

    def incorporateFeedback(self, state, action, reward, newState):
        raise NotImplementedError("Override me")

class QLearningAlgorithm(ValueLearningAlgorithm):
    """
    :description: Class implementing the Q-learning algorithm
    """
    def __init__(self, actions, discount, explorationProb, stepSize,
            minExplorationProb=.01, maxSteps=100000):
        super(QLearningAlgorithm, self).__init__(actions, discount, explorationProb, stepSize)
        self.steps = 0
        self.minExplorationProb = minExplorationProb
        self.maxSteps = maxSteps
        self.initialExplorationProb = explorationProb

    def incorporateFeedback(self, state, action, reward, newState):
        """
        :description: performs a Q-learning update

        :type state: dictionary
        :param state: the state of the game

        :type action: int
        :param action: the action for which to retrieve the Q-value

        :type reward: float
        :param reward: reward associated with being in newState

        :type newState: dictionary
        :param newState: the new state of the game

        :type rval: int or None
        :param rval: if rval returned, then this is the next action taken
        """
        self.steps += 1
        if self.explorationProb > 0.01:
            self.explorationProb = self.initialExplorationProb * (
                1 - self.steps / float(self.maxSteps))
            self.explorationProb = max(self.minExplorationProb, self.explorationProb)
            print('explorationProb: {}'.format(self.explorationProb))

        stepSize = self.stepSize
        prediction = self.getQ(state, action)
        target = reward
        if newState != None:
            target += self.discount * max(self.getQ(newState, newAction) 
                for newAction in self.actions)

        loss = 0
        for f, v in self.feature_extractor(state, action):
            diff = target - prediction
            loss += diff ** 2
            self.weights[f] = self.weights[f] + stepSize * (diff) * v

        return loss