import environment
import torch
import network
import numpy as np

gym = environment.ConnectXGym()
net = network.ConnectXNetwork(gym.positions.n, gym.actions.n)

gamma = 0.99
copy_step = 25
max_exp = 10000
min_exp = 100
batch_size = 32
learning_rate = 1e-2
epsilon = 0.5
decay = 0.9999
min_epsilon = 0.1
episodes = 200000

precision = 7

gym = environment.ConnectXGym()
num_states = gym.positions.n+1
print("num_states", num_states)
num_actions = gym.actions.n
TrainNet = environment.DQN(num_states, num_actions, gamma, max_exp, min_exp, batch_size, learning_rate)
TargetNet = environment.DQN(num_states, num_actions, gamma, max_exp, min_exp, batch_size, learning_rate)
observations = gym.reset()
print(TrainNet.model)
done =  False
while not done:
    action = TrainNet.get_action(observations, epsilon)

    prev_observations = observations

    observations, reward, done, _ = gym.step(action)

    if done:
        print(reward)
        gym.render()
        break
    
    
'''
observation = gym.trainer.reset()
board = np.array(observation.board)
board = torch.from_numpy(board).float()

while not gym.env.done:
    value, my_action = torch.max(net.forward(board).float(),0)
    observation, reward, done, info = gym.trainer.step(my_action.item())
    board = torch.from_numpy(np.array(observation.board)).float()
    gym.env.render()
'''

