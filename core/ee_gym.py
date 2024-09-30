import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import simpy

# Custom Gym Environment
class EarlyExitGym(gym.Env):
    def __init__(self, simpy_env):
        super(EarlyExitGym, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.state = 5.0
        self.deferred_rewards = []

    def reset(self):
        self.state = 5.0
        return np.array([self.state], dtype=np.float32)

    def step(self, item):
        self.state = item
        if self.state >= 20:

        return self.state, reward, False, {}
    def render(self, mode='human'):
        print(f"State: {self.state}")

    def close(self):
        pass

def simpy_process(simpy_env, gym_env, agent):
    obs = gym_env.reset()
    while True:
        yield simpy_env.timeout(1)
        action = agent.select_action(obs)
        next_obs, reward_deferred, done, _ = gym_env.step(action)
        reward = yield reward_deferred
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        agent.train()
        obs = next_obs
        gym_env.render()
        if done:
            obs = gym_env.reset()

def main():
    state_dim = 1
    action_dim = 2
    simpy_env = simpy.Environment()
    gym_env = BasicNonEpisodicEnv(simpy_env)
    agent = DQLAgent(state_dim, action_dim)
    simpy_env.process(simpy_process(simpy_env, gym_env, agent))
    simpy_env.run(until=100000)

if __name__ == "__main__":
    main()