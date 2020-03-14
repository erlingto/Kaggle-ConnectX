import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from kaggle_environments import evaluate, make

class ConnectXNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(ConnectXNetwork, self).__init__()
        self.conv1 = nn.Conv1d(6, 16, 1)
        self.conv2 = nn.Conv1d(16, 16, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_actions)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    
    



