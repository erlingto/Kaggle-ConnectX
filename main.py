import environment
import torch
import network
import numpy as np
import IPython
import help_func
import DQN2
import agents
from kaggle_environments import evaluate, make, utils

gym = environment.ConnectXGym()

gamma = 0.99
copy_step = 25
max_exp = 10000
min_exp = 100
batch_size = 100
learning_rate = 1e-2
epsilon = 0.5
decay = 0.9999
min_epsilon = 0.1
episodes = 200000


precision = 7
'''
TrainNet = environment.DQN(gym.positions.n, gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
TargetNet = environment.DQN(gym.positions.n, gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)

help_func.dojo(20000, gym, TrainNet, TargetNet, min_epsilon, epsilon, copy_step)
TrainNet.save_weights('trainvsselfmodel1')
'''
'''
Net = DQN2.DQN2(gym.positions.n, gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
Net.load_weights('lookahead_vs_verticalbot1')
print(Net)
print("paraneters:", len(list(Net.model.parameters())))
help_func.CreateFlippedAgent(Net)
'''
'''
template_gym = environment.ConnectXGym()
Opponent = DQN2.DQN2(gym.positions.n, gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
Opponent.load_weights('lookahead_vs_verticalbot1')
import help_func
help_func.playversusFlipped(Opponent)
'''
import sys
out = sys.stdout
submission = utils.read_file("submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
