
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'latent_hrl')
sys.path.append(os.path.abspath(path))

import envs.hallway as hallway

class TestHallway(unittest.TestCase):

    def test_step(self):
        env = hallway.HallwayEnv()
        num_steps = 1000
        for step in range(num_steps):
            a = np.random.randint(2)
            nx, r, d, _ = env.step(a)
            if d:
                nx = env.reset()
            print(a,nx,r,d)

if __name__ == '__main__':
    unittest.main()