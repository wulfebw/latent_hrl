
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random

class HallwayEnv(gym.Env):
    def __init__(self, length=5):
        self.length = length
        self.action_space = spaces.Discrete(2)
        low = np.array([1,0])
        high = np.array([5,1])
        self.observation_space = spaces.Box(low,high)
        self.goal_reward = 1
        self._reset()

    def _step(self, action):
        assert self.action_space.contains(action)
        x, g = self.state
        x += -1 if action == 0 else 1

        r, done = 0, False

        if (x == 1 and g == 0) or (x == self.length and g == 1):
            r = self.goal_reward
            done = True
        elif (x == 1 and g == 1) or (x == self.length and g == 0):
            done = True
            
        self.state = np.array([x, g])
        return self.state, r, done, {}

    def _reset(self):
        self.state = np.array([int(self.length / 2 + 1), random.randint(0, 1)])
        return self.state
