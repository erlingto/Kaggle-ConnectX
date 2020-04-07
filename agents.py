from kaggle_environments import evaluate, make, utils
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.optim as optim
import network
import torch.nn as nn
import torch
import math
import environment
import DQN2

class verticalBot():
    def __init__(self, num_columns, num_rows):
        self.states = num_columns * num_rows
        self.columns = num_columns
        self.rows = num_rows
        self.list_of_actions = [i for i in range(num_rows)]
        self.first_action = math.floor(np.random.random() * self.rows)

    def get_action(self, mark, board):
        count = 0
        possible_actions = self.list_of_actions
        position_up = board[self.first_action] + self.columns*(self.rows-1)
        count += 1
        for i in range(self.rows):
            if board[position_up] == mark or board[position_up] == 0:
                count +=1
                position_up = position_up - 1 * self.columns
            else:
                count = 0
        if count > 3:
            action = self.first_action
            return action
        else:
            self.first_action = math.floor(np.random.random() * self.rows)
            action = get_action(self, mark, board)
            return action

a = verticalBot(6, 7)
gym = DQN2.ConnectXGym2(a)

action = a.get_action(1, gym.env.board)
gym.step(action, 1)
gym.render()