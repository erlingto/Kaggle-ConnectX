from kaggle_environments import evaluate, make, utils
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import network
import math


class verticalBot():
    def __init__(self, num_columns, num_rows, mark = 1):
        self.states = num_columns * num_rows
        self.columns = num_columns
        self.rows = num_rows
        self.mark = mark
        self.name = "verticalbot"
        self.action = 0
        self.possible_actions =  [i for i in range(self.columns)]
        self.list_of_actions = [i for i in range(self.columns)]
    

    def get_action(self, observations, epsilon_placeholder= 0):
        mark = self.mark
        board = observations
        first = True
        bunnsum = 0
        for i in range(len(board)):
            bunnsum += board[i]
            if bunnsum > 2:
                first = False
                break
        
        if first:
            self.possible_actions =  [i for i in range(self.columns)]
            self.list_of_actions = [i for i in range(self.columns)]
            self.action = np.random.choice(self.list_of_actions)
            

        while True:
            if self.list_of_actions:
                count = 0
                position_up = self.action + self.columns*(self.rows-1)
                for i in range(self.rows):
                    if board[position_up] == mark or board[position_up] == 0:
                        count +=1
                    else:
                        count = 0
                    position_up = position_up - 1 * self.columns
                    
                if count > 3:
                    action = self.action
                    return action
                else:
                    self.list_of_actions.remove(self.action)
                    if self.list_of_actions:
                        self.action = np.random.choice(self.list_of_actions)
            else:
                
                action = np.random.choice(self.possible_actions)
                while board[action] != 0:
                    self.possible_actions.remove(action)
                    action = np.random.choice(self.possible_actions)
                return action
    

