
import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir, 'latent_hrl')
sys.path.append(os.path.abspath(path))

import envs.hallway
import rl.scratch

class RunningAvg(object):
    def __init__(self, value=0, rate=.99):
        self.value = value
        self.rate = rate
    def update(self, new_value):
        self.value = self.value * self.rate + (1 - self.rate) * new_value
    def __repr__(self):
        return str(self.value)

def simulate(env, agent, max_steps):
    x = env.reset()
    avg_r, avg_t, last_t = RunningAvg(), RunningAvg(), 0
    for step in range(max_steps):
        a = agent.step(x)
        print('agent action: {}'.format(a))
        print(agent.levels[-1].weights)
        nx, r, done, _ = env.step(a)
        agent.incorporate_feedback(r, done)
        print(x, a, r, nx, done)
        # raw_input()
        if done:
            avg_r.update(r)
            avg_t.update(step - last_t)
            print(step, avg_r, avg_t)
            last_t = step
            x = env.reset()
        else:
            x = nx

def run():
    np.random.seed(1)
    env = envs.hallway.HallwayEnv(length=9)
    actions = range(2)
    discount = .99
    explorationProb = .3
    stepSize = .1
    agent = rl.scratch.ScratchAlgorithm(actions, discount, explorationProb, 
        stepSize, max_k=5)
    simulate(env, agent, max_steps=10000000)




if __name__ == '__main__':
    run()