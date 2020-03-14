import environment
import torch
import network
import numpy as np

gym = environment.ConnectXGym()
net = network.ConnectXNetwork(gym.positions.n, gym.actions.n)

observation = gym.trainer.reset()
board = np.matrix(observation.board)
board = board.reshape(gym.columns, gym.rows)
board = torch.from_numpy(board)
board.unsqueeze_(-1)
print(board)



while not gym.env.done:
    my_action = max(net.forward(board))
    observation, reward, done, info = gym.trainer.step(my_action)
    board = torch.from_numpy(np.matrix(observation.board))
    gym.env.render()
