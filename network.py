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
        self.fc1 = nn.Linear(num_states, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_actions)
    
    def forward(self, x):   
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x 
    
    



