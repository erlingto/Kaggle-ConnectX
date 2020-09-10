from kaggle_environments import evaluate, make, utils
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.optim as optim
import network
import torch.nn as nn
import torch
import game_environment
import agents

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
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    return plt

class OpponentDQN:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.model = network.ConnectXNetwork2(num_states, num_actions)
        self.mark = 6
        self.name = 0
        self.EVALenv = game_environment.ConnectXEnvironment(7, 6, 4)
        
    def predict(self, inputs):
        return self.model(torch.from_numpy(inputs).float())
    
    def preprocess(self, state):
        result = state[:]
        return result

    def lookahead(self, state, action, mark, depth = 2):
        self.EVALenv.copy_board(state)
        new_state, valid, done, reward = self.EVALenv.step(action, mark)
        mark = mark % 2 +1
        if done:
            #print("DONE")
            value = reward[mark-1]
            #print("value", value)
            return value  
        elif depth == 0:
            prediction = self.predict(np.atleast_2d(self.preprocess(new_state)))[0].detach().numpy()
            value = max(prediction)
            if value > 1:
                value = 0.999
            elif value < -1:
                value = -0.999
            return value
        else:
            value = -1e7
            possible_actions = [i for i in range(self.num_actions) if new_state[i] == 0]
            for action in possible_actions:
                value = max(value, -self.lookahead(new_state, action, mark, depth -1))
        return value

    def get_action(self, state, epsilon):
        mark = 1
        if np.random.random() < epsilon:
            return(int(np.random.choice([c for c in range(self.num_actions) if state[c] == 0])))
        else:
            best_value = -1e7   
            best_action = 20 #want the shit to crash if an action isnt selected through negamax
            possible_actions = [i for i in range(self.num_actions) if state[i] == 0]
            for action in possible_actions:
                value = -self.lookahead(state, action, mark)
        
                if value > best_value:
                    best_action = action
                    best_value = value
            return best_action
    
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.name = path



class DQN:
    def __init__(self, num_states, num_actions, gamma, max_exp, min_exp, batch_size, learning_rate):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = network.ConnectXNetwork2(num_states, num_actions)
        self.optimizer = optim.Adam(self.model.parameters() ,lr = learning_rate)
        self.criterion = nn.MSELoss()
        self.mark = 1   #placeholder for verticalbot funticonality
        self.name = 0   
        self.EVALenv = game_environment.ConnectXEnvironment(7, 6, 4)
        
        self.experience = {'prev_obs' : [], 'a' : [], 'r': [], 'obs' : [], 'done': [] } 

        self.max_exp = max_exp
        self.min_exp = min_exp

    def predict(self, inputs):
        return self.model(torch.from_numpy(inputs).float())
    
    
    def preprocess(self, state):
        result = state[:]
        return result

    def get_action_no_lookahead(self, state, epsilon):
        if np.random.random() < epsilon:
            return(int(np.random.choice([c for c in range(self.num_actions) if state[c] == 0])))
        else:
            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().numpy()
            for i in range(self.num_actions):
                if state[i] != 0:
                    prediction[i] = -1e7
            return int(np.argmax(prediction))  

    
    def lookahead(self, state, action, mark, alpha, beta, depth = 2):
        self.EVALenv.copy_board(state)
        new_state, valid, done, reward = self.EVALenv.step(action, mark)
        mark = mark % 2 +1
        if done:
            value = reward[mark-1]
            return value  
        elif depth == 0:
            prediction = self.predict(np.atleast_2d(self.preprocess(new_state)))[0].detach().numpy()
            value = max(prediction)
            
            if value > 20:
                value = 19.999
            elif value < -20:
                value = -19.999
            return value
        else:
            value = -1e7
            possible_actions = [i for i in range(self.num_actions) if new_state[i] == 0]
            for action in possible_actions:
                value = max(value, -self.lookahead(new_state, action, mark, -beta, -alpha, depth -1))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        return value

    def get_action(self, state, epsilon):
        mark = 1
        if np.random.random() < epsilon:
            #print("Random")
            return(int(np.random.choice([c for c in range(self.num_actions) if state[c] == 0])))
        else:
            alpha = float("-inf")
            beta = float("inf")
            best_value = -1e7
            best_action = 20 #want the shit to crash if an action isnt selected through negamax
            possible_actions = [i for i in range(self.num_actions) if state[i] == 0]
            for action in possible_actions:
                value = -self.lookahead(state, action, mark, -beta, -alpha)
        
                if value > best_value:
                    best_action = action
                    best_value = value
            return best_action
    
    def get_values(self, state, TargetNet):
        alpha = float("-inf")
        beta = float("inf")
        mark = 1
        best_value = -1e7
        best_action = 20 #want the shit to crash if an action isnt selected through negamax
        possible_actions = [i for i in range(self.num_actions) if state[i] == 0]
        value = TargetNet.predict(state).detach().numpy()
        for i in range(len(value)):
            if value[i] > 20:
                    value[i] = 19.999
            elif value[i] < -20:
                    value[i] = -19.999
        for action in possible_actions:
            value[action] = -self.lookahead_values(state, action, mark, -beta, -alpha, TargetNet )
        return value

    def lookahead_values(self, state, action, mark, alpha, beta, TargetNet, depth = 1):
        self.EVALenv.copy_board(state)
        new_state, valid, done, reward = self.EVALenv.step(action, mark)
        mark = mark % 2 +1
        if done:
            value = reward[mark-1]
            return value  
        elif depth == 0:
            prediction = TargetNet.predict(np.atleast_2d(self.preprocess(new_state)))[0].detach().numpy()
            value = max(prediction)
            if value > 20:
                value = 19.999
            elif value < -20:
                value = -19.999
            return value
        else:
            value = -1e7
            possible_actions = [i for i in range(self.num_actions) if new_state[i] == 0]
            for action in possible_actions:
                value = max(value, -self.lookahead_values(new_state, action, mark, -beta, -alpha, TargetNet, depth -1))
                alpha = max(alpha, value)
                if alpha >= beta:
                    
                    break
        return value

    
    def add_experience(self, exp):
        if len(self.experience['prev_obs']) >= self.max_exp:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.name = path

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
    
    def copy_weights(self, TrainNet):
        self.model.load_state_dict(TrainNet.model.state_dict())

    def train(self, TargetNet):
        if len(self.experience['prev_obs']) < self.min_exp:
            return 0
        
        ids =  np.random.randint(low = 0, high = len(self.experience['prev_obs']), size = self.batch_size)
        states = np.asarray([self.preprocess(self.experience['prev_obs'][i]) for i in ids])
        actions  = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        next_states = np.asarray([self.preprocess(self.experience['obs'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])       
        next_value = np.max(TargetNet.predict(next_states).detach().numpy(), axis=1)
        """
        next_value = np.zeros(self.batch_size)
        k = 0

        for next_state in next_states:
            next_value[k] = np.max(self.get_values(next_state, TargetNet))
            k+=1
        """

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



class ConnectXGym2(gym.Env):
    def __init__(self):
        self.env = game_environment.ConnectXEnvironment(7, 6, 4)
    
        self.trainer = 0
        
      
        self.columns = self.env.num_columns
        self.rows = self.env.num_rows
        self.actions = gym.spaces.Discrete(self.columns)
        self.positions = gym.spaces.Discrete(self.columns * self.rows)
        self.list_of_trainers = ["new2", "new", "new1"]
        self.score_list = {i : 0 for i in self.list_of_trainers}
        self.games_list = {i : 0 for i in self.list_of_trainers}
        self.change_trainer_at_random()
    
    def reset_scores(self):
        self.score_list = {i : 0 for i in self.list_of_trainers}
        self.games_list = {i : 0 for i in self.list_of_trainers}
    def print_scores(self):
        for i in self.score_list:
            print(i, self.score_list[i], end = "")
            print("/", self.games_list[i])
    
    def change_trainer(self, trainer):
        self.trainer = trainer

    def change_trainer_at_random(self):
        choice = np.random.choice(self.list_of_trainers)
        if choice == "verticalbot":
            trainer = agents.verticalBot(7,6)
            self.change_trainer(trainer)
        else:
            trainer = OpponentDQN(42,7)
            trainer.load_weights(choice)
            self.change_trainer(trainer)
        

    def step(self, action, mark):
        return self.env.step(action, mark)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def generate_data(self, TrainNet, TargetNet, epsilon, copy_step):
        rewards = 0
        opp_action = 0
        done = False

        env = self.env
        
        trainee_mark, observations = env.reset()
        
        if trainee_mark == 1:
            TrainNet.mark = 1
            self.trainer.mark = 2
            opp_mark = 2
                 
            while not done:
                '''trainee makes a move '''
                action = TrainNet.get_action(observations, epsilon)

                prev_observations = np.array(observations)
        
                observations, valid, done, reward = env.step(action, trainee_mark)

                reward = reward[trainee_mark-1]
                if not done:
                    ''' opponent makes a move '''
                    ''' flip_the board '''
                    observations = env.flip()
                    opp_action = self.trainer.get_action(observations, 0.03)

                    observations, valid, done, reward = env.step(opp_action, opp_mark)
                    reward = reward[trainee_mark-1]

                if done:
                    rewards += reward

                exp = {'prev_obs': prev_observations, 'a' : action, 'r': reward, 'obs': observations, 'done' : done }
                TrainNet.add_experience(exp)

                loss = TrainNet.train(TargetNet)

            return rewards, loss
        else:
            TrainNet.mark = 2
            self.trainer.mark = 1
            opp_mark = 1     
            ''' opponent makes a move '''
            opp_action = self.trainer.get_action(observations, 0.03)

            observations, valid, done, reward = env.step(opp_action, opp_mark)
            reward = reward[trainee_mark-1]

            while not done:
                observations = env.flip()
                action = TrainNet.get_action(observations, epsilon)

                prev_observations = np.array(observations)
        
                observations, valid, done, reward = env.step(action, trainee_mark)
                reward = reward[trainee_mark-1]
                if not done:
                    ''' opponent makes a move '''
                    opp_action = self.trainer.get_action(observations, 0.03)

                    observations, valid, done, reward = env.step(opp_action, opp_mark)
                    reward = reward[trainee_mark-1]

                if done:
                    rewards += reward

                exp = {'prev_obs': prev_observations, 'a' : action, 'r': reward, 'obs': env.flip(), 'done' : done }
                TrainNet.add_experience(exp)

                loss = TrainNet.train(TargetNet)
            return rewards, loss

def dojo(games, gym, TrainNet, TargetNet, min_epsilon, epsilon, copy_step):
    total_loss = 0
    even_match = 0
    test_match = game_environment.ConnectXEnvironment(7, 6, 4)
    _, test_state = test_match.reset()
    print(TrainNet.predict(test_state))    
    decay = 0.9995
    for i in range(games):
        rewards, loss = gym.generate_data(TrainNet, TargetNet, epsilon, copy_step)
        if rewards == 0:
            even_match += 1
        gym.render()
        print("motstander", gym.trainer.mark)
        print("SCORE:", rewards)
        gym.score_list[gym.trainer.name] += rewards
        gym.games_list[gym.trainer.name] += 1
        total_loss += loss
        print(i)
        if i%10 == 0 and i != 0:
            gym.change_trainer_at_random()
            print(TrainNet.predict(test_state))  
        if i%2 == 0 and i !=  0:
            epsilon = max(min_epsilon, epsilon*decay)
        if i%100 == 0 and i != 0:
            print('Total Loss:', total_loss)
            print('Even matches:', even_match)
            gym.print_scores()
            gym.reset_scores()
            even_match = 0
            total_loss = 0
            print("games", i)
            print("epsilon", epsilon)
        if i % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
        if i%50000 == 0 and i != 0:
            plt = plot_grad_flow(TrainNet.model.named_parameters())
            path = "plot" + str(i)+ ".png"
            plt.savefig(path)

