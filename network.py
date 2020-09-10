import torch.nn as nn
import torch

class ConnectXNetwork2(nn.Module):
    def __init__(self, num_states, num_actions):
        super(ConnectXNetwork2, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128 , num_actions)
    
    def forward(self, x):   
        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = nn.functional.leaky_relu(self.fc3(x))
        x = nn.functional.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x 