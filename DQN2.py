from kaggle_environments import evaluate, make, utils
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.optim as optim
import network
import torch.nn as nn
import torch
import environment

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    return plt


class ConnectXNetwork2(nn.Module):
    def __init__(self, num_states, num_actions):
        super(ConnectXNetwork2, self).__init__()
        self.fc1 = nn.Linear(num_states+1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, num_actions)
    
    def forward(self, x):   
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return x 

    
class DQN2:
    def __init__(self, num_states, num_actions, gamma, max_exp, min_exp, batch_size, learning_rate):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = ConnectXNetwork2(num_states, num_actions)
        self.optimizer = optim.Adam(self.model.parameters() ,lr = learning_rate)
        self.criterion = nn.MSELoss()
        
        self.experience = {'prev_obs' : [], 'a' : [], 'r': [], 'obs' : [], 'done': [] } 

        self.max_exp = max_exp
        self.min_exp = min_exp

    def predict(self, inputs):
        return self.model(torch.from_numpy(inputs).float())
    
    def preprocess(self, state):
        result = state.board[:]
        result = np.append(result, state.mark)
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
        return loss

class Observation:
    def __init__(self, board, mark):
        self.board = board
        self.mark = mark

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
        j = 0
        for i in range(self.num_columns):
            if not self.board[i] == 0:
                j +=1
                if j == self.num_columns:
                    done = True
                    reward = [0.5, 0.5]
                    return done, reward


        mark = self.board[position]
        reward = [0, 0]
        done = 0

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
                        done = True
                        return done, reward

        return done, reward


    def step(self, action, mark):
        reward = [0.5, 0.5]
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
                observation = Observation(self.board, mark)
                
                return observation, valid, done, reward
            else:
                print("action is full")
                valid = False
                observation = Observation(self.board, mark)
                return observation, valid, done, reward
        else:
            print("action : ", action, end = '')
            print("is out of bounds")
            valid = False
            observation = Observation(self.board, mark)
            return observation, valid, done, reward
    

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
        self.board = np.zeros(self.num_columns* self.num_rows, dtype = int)
        self.marks = [1, 2]
        self.done = False

        coinflip = np.random.random()
        
        if coinflip < 0.5:
            trainee_mark = 1

        else:
            trainee_mark = 2
    

        observations = Observation(self.board, trainee_mark)
        return observations

    

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
    def __init__(self, trainer):
        self.env = ConnectXEnvironment(7, 6, 4)
    
        self.trainer = trainer
      
        self.columns = self.env.num_columns
        self.rows = self.env.num_rows
        self.actions = gym.spaces.Discrete(self.columns)
        self.positions = gym.spaces.Discrete(self.columns * self.rows)

    def step(self, action, mark):
        return self.env.step(action, mark)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def generate_data(self, TrainNet, TargetNet, epsilon, copy_step):
        rewards = 0
        iter = 0
        opp_action = 0
        done = False

        env = self.env
        
        observations = env.reset()
        
        if observations.mark == 1:
            opp_mark = 2
            trainee_mark = 1     

            while not done:
                action = TrainNet.get_action(observations, epsilon)

                prev_observations = observations
        
                observations, valid, done, reward = env.step(action, trainee_mark)
                reward = reward[trainee_mark-1]
                if not done:
                    ''' opponent makes a move '''
                    opp_action = self.trainer.get_action(observations, 0)

                    observations, valid, done, reward = env.step(opp_action, opp_mark)
                    reward = reward[trainee_mark-1]

                if done:
                    if reward == 1:
                        reward = 1
                    elif reward  == 0:
                        reward = -1
                    else :
                        reward = 0
                        env.render()
                   
                    rewards += reward

                exp = {'prev_obs': prev_observations, 'a' : action, 'r': reward, 'obs': observations, 'done' : done }
                TrainNet.add_experience(exp)

                loss = TrainNet.train(TargetNet)
                iter += 1
                if iter % copy_step == 0:
                    TargetNet.copy_weights(TrainNet)
            return rewards, loss
        else:
            opp_mark = 1
            trainee_mark = 2     
            
            opp_action = self.trainer.get_action(observations, 0)

            observations, valid, done, reward = env.step(opp_action, opp_mark)
            reward = reward[trainee_mark-1]

            while not done:
                action = TrainNet.get_action(observations, epsilon)

                prev_observations = observations
        
                observations, valid, done, reward = env.step(action, trainee_mark)
                reward = reward[trainee_mark-1]
                if not done:
                    ''' opponent makes a move '''
                    opp_action = self.trainer.get_action(observations, 0)

                    observations, valid, done, reward = env.step(opp_action, opp_mark)
                    reward = reward[trainee_mark-1]

                if done:
                    if reward == 1:
                        reward = 1
                    elif reward  == 0:
                        reward = -1
                    else :
                        reward = 0
                        env.render()
                    rewards += reward

                exp = {'prev_obs': prev_observations, 'a' : action, 'r': reward, 'obs': observations, 'done' : done }
                TrainNet.add_experience(exp)

                loss = TrainNet.train(TargetNet)
                iter += 1
                if iter % copy_step == 0:
                    TargetNet.copy_weights(TrainNet)
            return rewards, loss

def dojo(games, gym, TrainNet, TargetNet, min_epsilon, epsilon, copy_step):
    total_reward = 0
    total_loss = 0
    decay = 0.999
    for i in range(games):
        rewards, loss = gym.generate_data(TrainNet, TargetNet, epsilon, copy_step)
        total_reward += rewards
        total_loss += loss
        if i%300 == 0 and i is not 0:
            epsilon = max(min_epsilon, epsilon*decay)
        if i%1000 == 0 and i is not 0:
            print('Total Reward:', total_reward)
            print('Total Loss:',total_loss)
            total_reward = 0
            total_loss = 0
            print(epsilon)
            plt = plot_grad_flow(TrainNet.model.named_parameters())
            path = "plot" + str(i)+ ".png"
            plt.savefig(path)

'''


gamma = 0.99
copy_step = 25
max_exp = 100000
min_exp = 100
batch_size = 100
learning_rate = 0.00146
epsilon = 0.5
decay = 0.999999999999999
min_epsilon = 0.05
episodes = 200000

precision = 7
template_gym = environment.ConnectXGym()
Opponent = DQN2(template_gym.positions.n, template_gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)

TrainNet = DQN2(template_gym.positions.n, template_gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
TargetNet = DQN2(template_gym.positions.n, template_gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)

training_gym = ConnectXGym2(Opponent)



dojo(50000, training_gym, TrainNet, TargetNet, min_epsilon, epsilon, copy_step)
Opponent = DQN2(template_gym.positions.n, template_gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
training_gym = ConnectXGym2(Opponent)
TrainNet.save_weights('fivenet1.0')
Opponent = DQN2(template_gym.positions.n, template_gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
Opponent.load_weights('fivenet1.0')
training_gym = ConnectXGym2(Opponent)
dojo(50000, training_gym, TrainNet, TargetNet, min_epsilon, epsilon, copy_step)
TrainNet.save_weights('fivenet1.0')

Opponent = DQN2(template_gym.positions.n, template_gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
Opponent.load_weights('fivenet1.0')
training_gym = ConnectXGym2(Opponent)
dojo(50000, training_gym, TrainNet, TargetNet, min_epsilon, epsilon, copy_step)
TrainNet.save_weights('fivenet1.0')
Opponent = DQN2(template_gym.positions.n, template_gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
Opponent.load_weights('fivenet1.0')
training_gym = ConnectXGym2(Opponent)
dojo(50000, training_gym, TrainNet, TargetNet, min_epsilon, epsilon, copy_step)
TrainNet.save_weights('fivenet4.0')

template_gym = environment.ConnectXGym()
Opponent = DQN2(template_gym.positions.n, template_gym.actions.n, gamma, max_exp, min_exp, batch_size, learning_rate)
Opponent.load_weights('fivenet1.0')
import help_func
help_func.playversus(Opponent)

'''
