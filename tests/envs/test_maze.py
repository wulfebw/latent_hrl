
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'latent_hrl')
sys.path.append(os.path.abspath(path))

import envs.maze

class TestHallway(unittest.TestCase):

    def test_step(self):
        maze = envs.maze.MazeEnv(room_length=3, num_rooms_per_side=2)
        # up until wall
        a = 0
        maze.state = np.array([0, 0])
        nx, _, _, _ = maze.step(a)
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(nx, [0,2])
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(nx, [0,2])
        
        # down until wall
        a = 2
        maze.state = np.array([0, 2])
        nx, _, _, _ = maze.step(a)
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(nx, [0,0])
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(nx, [0,0])

        # right until wall
        maze.state = np.array([0, 0])
        a = 1
        nx, _, _, _ = maze.step(a)
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(nx, [2,0])
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(nx, [2,0])

        # left until wall
        maze.state = np.array([2, 0])
        a = 3
        nx, _, _, _ = maze.step(a)
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(nx, [0,0])
        nx, _, _, _ = maze.step(a)
        np.testing.assert_array_equal(nx, [0,0])

        # through doorway to the right until wall
        maze.state = np.array([0, 0])
        nx, _, _, _ = maze.step(0) # up 
        nx, _, _, _ = maze.step(1) # right
        nx, _, _, _ = maze.step(1) # right
        nx, _, _, _ = maze.step(1) # right
        nx, _, _, _ = maze.step(1) # right
        nx, _, _, _ = maze.step(1) # right
        np.testing.assert_array_equal(nx, [5,1])
        nx, _, _, _ = maze.step(1) # right
        np.testing.assert_array_equal(nx, [5,1])

        # back through the doorway I came, and then up through the other doorway
        maze.state = np.array([5, 1])
        nx, _, _, _ = maze.step(3) # left
        nx, _, _, _ = maze.step(3) # left
        nx, _, _, _ = maze.step(3) # left
        nx, _, _, _ = maze.step(3) # left
        nx, _, _, _ = maze.step(0) # up
        nx, _, _, _ = maze.step(0) # up
        nx, _, _, _ = maze.step(0) # up
        nx, _, _, _ = maze.step(0) # up
        np.testing.assert_array_equal(nx, [1,5])
        nx, _, _, _ = maze.step(0) # up
        np.testing.assert_array_equal(nx, [1,5])

        # to the goal state
        maze.state = np.array([1, 5])
        nx, _, _, _ = maze.step(1) # right
        nx, _, _, _ = maze.step(2) # down
        nx, _, _, _ = maze.step(1) # right
        nx, _, _, _ = maze.step(1) # right
        nx, _, _, _ = maze.step(1) # right
        nx, _, _, _ = maze.step(0) # up
        np.testing.assert_array_equal(nx, [5,5])

        # down until wall
        maze.state = np.array([5, 5])
        nx, _, _, _ = maze.step(2) # down
        nx, _, _, _ = maze.step(2) # down
        np.testing.assert_array_equal(nx, [5,3])
        nx, _, _, _ = maze.step(2) # down
        np.testing.assert_array_equal(nx, [5,3])

if __name__ == '__main__':
    unittest.main()