
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random

class MazeEnv(gym.Env):
    def __init__(self, room_length=3, num_rooms_per_side=2):
        assert room_length % 2 == 1, "room_length must be odd"
        assert room_length >= 3, "room_length must be greater than 3"
        assert num_rooms_per_side >= 1, "must have at least 1 room"

        self.room_length = room_length
        self.num_rooms_per_side = num_rooms_per_side
        # 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space = spaces.Discrete(4)
        low = np.array([0,0])
        self.max_pos = room_length * num_rooms_per_side - 1
        high = np.array([self.max_pos, self.max_pos])
        self.observation_space = spaces.Box(low,high)
        self.goal_reward = 1
        self.goal_state = [self.max_pos, self.max_pos]
        self._reset()

    def _reset(self):
        self.state = [0,0]
        return self.state

    def _step(self, a):
        assert self.action_space.contains(a)
        x, y = self.state

        # up
        if a == 0:
            y = self._step_up(x, y)
        # right
        elif a == 1:
            x = self._step_right(x, y)
        # down
        elif a == 2:
            y = self._step_down(x, y)
        # left
        else:
            x = self._step_left(x, y)

        r, done = 0, False
        if x == self.goal_state[0] and y == self.goal_state[1]:
            r, done = self.goal_reward, True
            
        self.state = np.array([x, y])
        return self.state, r, done, {}

    def _step_up(self, x, y):
        ny = y + 1

        # convert to single room format
        local_ny = ny % self.room_length

        # this condition True indicates passing through wall
        if local_ny == 0:

            # this is only allowed if passing through doorway
            if not (x % self.room_length == self.room_length // 2):
                ny = y

        ny = min(ny, self.max_pos)
        return ny

    def _step_right(self, x, y):
        nx = x + 1

        # convert to single room format
        local_nx = nx % self.room_length

        # this condition True indicates passing through wall
        if local_nx == 0:

            # this is only allowed if passing through doorway
            if not (y % self.room_length == self.room_length // 2):
                nx = x

        nx = min(nx, self.max_pos)
        return nx

    def _step_down(self, x, y):        
        ny = y - 1

        # convert to single room format
        local_ny = ny % self.room_length

        # this condition True indicates passing through wall
        if local_ny == self.room_length - 1:

            # this is only allowed if passing through doorway
            if not (x % self.room_length == self.room_length // 2):
                ny = y

        ny = max(0, ny)
        return ny

    def _step_left(self, x, y):
        nx = x - 1

        # convert to single room format
        local_nx = nx % self.room_length

        # this condition True indicates passing through wall
        if local_nx == self.room_length - 1:

            # this is only allowed if passing through doorway
            if not (y % self.room_length == self.room_length // 2):
                nx = x

        nx = max(0, nx)
        return nx