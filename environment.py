from kaggle_environments import evaluate, make, utils
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

class ConnectXGym(gym.Env):
    def __init__(self, switch_prob = 0.5):
        self.env = make('connectx', debug  = False)
        self.pair = [None, 'random']
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob


        config = self.env.configuration
        self.columns = config.columns
        self.rows = config.rows
        self.actions = gym.spaces.Discrete(config.columns)
        self.positions = gym.spaces.Discrete(config.columns * config.rows)

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)
    
    def step(self, action):
        return self.trainer.step(action)

    def reset(self):
        if np.random.random() > self.switch_prob:
            self.switch_trainer()
        return self.trainer.reset()
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)

