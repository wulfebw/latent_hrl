
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir, 'latent_hrl')
sys.path.append(os.path.abspath(path))

import envs.maze
import rl.algorithms

class RunningAvg(object):
    def __init__(self, value=0, rate=.99):
        self.value = value
        self.rate = rate
    def update(self, new_value):
        self.value = self.value * self.rate + (1 - self.rate) * new_value
    def __repr__(self):
        return str(self.value)

def train(env, agent, max_steps):
    x = env.reset()
    avg_r, avg_t, last_t = RunningAvg(), RunningAvg(), 0
    for step in range(max_steps):
        a = agent.step(x)
        nx, r, done, _ = env.step(a)

        if done:
            agent.incorporate_feedback(x, a, r, None)
            avg_t.update(step - last_t)
            avg_r.update(r * agent.discount ** (step - last_t))
            last_t = step
            x = env.reset()
            print(step, avg_r, avg_t)
        else:
            agent.incorporate_feedback(x, a, r, nx)
            x = nx

def collect(env, agent, max_steps):
    states, actions = [], []
    ep_states, ep_actions = [], []
    x = env.reset()
    ep_states.append(x)
    for step in range(max_steps):
        a = agent.step(x)
        ep_actions.append(a)
        nx, r, done, _ = env.step(a)
        ep_states.append(nx)
        if done:
            x = env.reset()
            states.append(ep_states)
            actions.append(ep_actions)
            ep_states, ep_actions = [], []
            ep_states.append(x)
        else:
            x = nx

    return np.asarray(states), np.asarray(actions)

def analyze(env, agent):
    states = np.array([[x,y] for x in range(env.max_pos + 1) 
        for y in range(env.max_pos + 1)])
    values = np.empty(states.shape[0])
    actions = np.empty(states.shape[0])

    for i, state in enumerate(states):
        max_value, max_action = max((agent.getQ(state, action), action) 
            for action in agent.actions)
        values[i] = max_value
        actions[i] = max_action

    values = np.rot90(values.reshape(env.max_pos + 1, env.max_pos + 1), 1)
    actions = np.rot90(actions.reshape(env.max_pos + 1, env.max_pos + 1), 1)
    plt.subplot('211')
    plt.imshow(values)
    plt.colorbar()
    plt.title('state values')
    plt.subplot('212')
    plt.imshow(actions)
    plt.colorbar()
    plt.title('policy')
    plt.tight_layout()
    plt.show()

def run():
    np.random.seed(1)
    env = envs.maze.MazeEnv(room_length=3, num_rooms_per_side=2)
    actions = range(4)
    discount = .95
    explorationProb = .7
    stepSize = .1
    max_steps = 1000000
    agent = rl.algorithms.QLearningAlgorithm(actions, discount, explorationProb, 
        stepSize, maxSteps=max_steps)
    train(env, agent, max_steps=max_steps)
    agent.explorationProb = 0.
    agent.minExplorationProb = 0.
    collect_steps = 1000000
    states, actions = collect(env, agent, collect_steps)
    output_filepath = '../data/runs/maze.npz'
    np.savez(output_filepath, states=states, actions=actions)
    analyze(env, agent)

if __name__ == '__main__':
    run()