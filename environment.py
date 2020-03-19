from kaggle_environments import evaluate, make, utils
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.optim as optim
import network
import torch.nn as nn
import torch


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


class DQN:
    def __init__(self, num_states, num_actions, gamma, max_exp, min_exp, batch_size, learning_rate):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = network.ConnectXNetwork(num_states, num_actions)
        self.optimizer = optim.Adam(self.model.parameters() ,lr = learning_rate)
        self.criterion = nn.MSELoss()
        
        self.experience = {'prev_obs' : [], 'a' : [], 'r': [], 'obs' : [], 'done': [] } 

        self.max_exp = max_exp
        self.min_exp = min_exp

    def predict(self, inputs):
        return self.model(torch.from_numpy(inputs).float())
    
    def preprocess(self, state):
        result = state.board[:]
        result.append(state.mark)
        return result

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return(int(np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0])))
        else:
            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().numpy()
            for i in range(self.num_actions):
                if state.board[i] != 0:
                    prediction[i] = -1e7
            return int(np.argmax(prediction))  
    
    def add_experience(self, exp):
        if len(self.experience['prev_obs']) >= self.max_exp:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
    
    def copy_weights(self, TrainNet):
        self.model.load_state_dict(TrainNet.state_dict())

    def train(self, TargetNet):
        if len(self.experience['prev_obs']) < self.min_exp:
            return 0
        
        ids =  np.random.randint(low = 0, high = len(self.experience['prev_obs']), size = self.batch_size)
        states = np.asarray([self.preprocess(self.experience['prev_obs'][i]) for i in ids])
        actions  = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])

        next_states = np.asarray([self.preprocess(self.experience['obs'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        next_value = np.max(TargetNet.predict(next_states).detach().numpy(), axis = 1)

        ''' Q - learning aspect '''
        actual_values = np.where(dones, rewards, rewards+self.gamma*next_value)
        '''  !!!    '''
        actions = np.expand_dims(actions, axis = 1)
        
        actions_one_hot = torch.FloatTensor(self.batch_size, self.num_actions).zero_()
        actions_one_hot = actions_one_hot.scatter_(1, torch.LongTensor(actions), 1)
        selected_action_values = torch.sum(self.predict(states) * actions_one_hot, dim = 1)
        actual_values = torch.FloatTensor(actual_values)
        
        
        self.optimizer.zero_grad()
        loss = self.criterion(selected_action_values, actual_values)
        loss.backward()
        self.optimizer.step()
        
   
class ConnectXEnvironment:
    def __init__(self, num_columns, num_rows, connect):
        self.connect = connect
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.size = num_rows * num_columns
        self.board = np.zeros(num_columns* num_rows, dtype = int)
        self.marks = [1, 2]
        self.done = 0
    
    def check(self, position):
        column = position % self.num_columns 
        row = int((position - column) / self.num_columns)
        for i in range(self.num_columns):
            if self.board[i] == 0:
                break
            else:
                done = True
                reward = [0.5, 0.5]


        mark = self.board[position]
        reward = [0, 0]
        done = 0

        print(row) 
        print(self.num_rows-self.connect)

        win_condition_up = 1
        win_condition_down = 1
        win_condition_dl = 1
        win_condition_dr = 1
        win_condition_ur = 1
        win_condition_ul = 1
        win_condition_r = 1
        win_condition_l = 1

        ''' right '''
        position_right = position
        position_diagonal_dr = position 
        position_diagonal_ur = position 
        ''' left '''
        position_left = position
        position_diagonal_dl = position 
        position_diagonal_ul = position
        ''' up and down '''
        position_down = position 
        position_up = position
        for i in range(self.connect-1):
            print("i", i)
            ''' right '''
            if column <= self.num_columns - self.connect:
                position_right = position_right + 1
                if self.board[position_right] == mark:
                    win_condition_r +=1
                    if win_condition_r == self.connect:
                        reward[mark - 1] = 1 
                        done = True
                        return done, reward
                if row <= self.num_rows - self.connect:
                    position_diagonal_dr = position_diagonal_dr + 1 * self.num_columns + 1
                    if self.board[position_diagonal_dr] == mark:
                        win_condition_dr +=1
                    if win_condition_dr == self.connect:
                        reward[mark - 1] = 1 
                        
                        done = True
                        return done, reward
                if row >= self.connect:
                    position_diagonal_ur = position_diagonal_ur - 1 * self.num_columns + 1
                    if self.board[position_diagonal_ur] == mark:
                        win_condition_ur +=1
                        
                    if win_condition_ur == self.connect:
                        
                        reward[mark - 1] = 1 
                        done = True
                        return done, reward 

            ''' left '''
            if column +1 >= self.connect:
                
                position_left = position_left - 1
                if self.board[position_left] == mark:
                    win_condition_l +=1
                    if win_condition_l == self.connect:
                       
                        reward[mark - 1] = 1 
                        done = True
                        return done, reward
                '''down'''    
                if row <= self.num_rows - self.connect:
                    position_diagonal_dl = position_diagonal_dl + 1  * self.num_columns - 1
                    if self.board[position_diagonal_dl] == mark:
                        win_condition_dl +=1
                        
                        if win_condition_dl == self.connect:
                            reward[mark - 1] = 1 
                            done = True
                            return done, reward
                ''' up '''
                if row >= self.connect:
                    position_diagonal_ul = position_diagonal_ul - 1 * self.num_columns - 1
                    if self.board[position_diagonal_ul] == mark:
                        win_condition_ul +=1
                        
                        if win_condition_ul == self.connect:
                            reward[mark - 1] = 1 
                            return done, reward
            ''' up and down ''' 
            if row <= self.num_rows-self.connect:
                position_down = position_down + 1  * self.num_columns
                if self.board[position_down] == mark:
                    win_condition_down +=1
                    if win_condition_down == self.connect:
                        
                        reward[mark - 1] = 1 
                        done = True
                        return done, reward

            if row >= self.connect:
                position_up = position_up - 1 * self.num_columns
                if self.board[position_up] == mark:
                    win_condition_up +=1
                    if win_condition_up == self.connect:
                       
                        reward[mark - 1] = 1 
                        done = 1
                        done = True
                        return done, reward

        return done, reward


    def step(self, action, mark):
        reward = [0,0]
        done = False
        valid = True
        if action < self.num_columns + 1:
            if self.board[action] == 0:
                k = 0
                while self.board[action + self.num_columns * k] == 0:
                        k += 1
                        if k == self.num_rows:
                            break
                self.board[action + self.num_columns * (k-1)] = mark
                done, reward = self.check(action + self.num_columns* (k-1))
                return self.board, valid, done, reward
            else:
                print("action is full")
                valid = False
                return self.board, valid, done, reward
        else:
            print("action : ", action, end = '')
            print("is out of bounds")
            valid = False
            return self.board, valid, done, reward
    

    def render(self):
        for k in range(self.num_columns * 4 +1):
                if k%4 == 0:
                    print('+', end = '')
                else:
                    print('-', end = '')
        print('\n', end = '')
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                print('|', ' ', sep = '', end = '')
                print(self.board[i*self.num_columns + j] ,' ', end = '', sep= '')
            print('| \n', end = '')
            for k in range(self.num_columns * 4 +1):
                if k%4 == 0:
                    print('+', end = '')
                else:
                    print('-', end = '')
            print('\n', end = '')
            
    def reset(self):
        self.connect = connect
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.size = num_rows * num_columns
        self.board = np.zeros(num_columns* num_rows, dtype = int)
        self.marks = [1, 2]
        self.done = False

    def play_game(self):
        done = False
        mark = 1
        
        while not done:
            print("player: ", mark)
            self.render()
            self.board, valid, done, reward = self.step(int(input("choose your action:")), mark)
            print("done:", done)
            if valid == True:
                mark = mark % 2
                mark += 1
        self.render()
            
                


class ConnectXGym2(gym.Env):
    def __init__(self, model, switch_prob = 0.5):
        self.env = ConnectXEnvironment(7, 6, 4)
        self.trainer = model
        self.switch_prob = switch_prob


        self.columns = self.env.num_columns
        self.rows = self.env.num_rows
        self.actions = gym.spaces.Discrete(self.columns)
        self.positions = gym.spaces.Discrete(self.columns * self.rows)

    def step(self, action, mark):
        return self.env.step(action, mark)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    