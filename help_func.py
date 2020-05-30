import environment
import types
import torch
import network
import numpy as np
import IPython
from kaggle_environments import evaluate, make, utils
import DQN2


def generate_data(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False

    observations = env.reset()

    while not done:
        action = TrainNet.get_action(observations, epsilon)

        prev_observations = observations
    
        observations, reward, done, _ = env.step(action)

        if done:
            if reward == 1:
                reward = 30
            elif reward == 0:
                reward = -30
            else :
                reward = 13
            rewards += reward

        exp = {'prev_obs': prev_observations, 'a' : action, 'r': reward, 'obs': observations, 'done' : done }
        TrainNet.add_experience(exp)

        loss = TrainNet.train(TargetNet)
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, loss

def dojo(games, env, TrainNet, TargetNet, min_epsilon, epsilon, copy_step):
    decay = 0.9999
    total_reward = 0
    total_loss = 0
    for i in range(games):
        epsilon = max(min_epsilon, epsilon*decay)
        reward, loss = generate_data(env, TrainNet, TargetNet, epsilon, copy_step)
        total_reward += reward
        total_loss += loss
        if i%100 == 0:
            print(total_reward)
            print(total_loss)
            total_loss = 0
            total_reward = 0

def CreateAgent(DQN):
    layers = []
    # Get layers' weights
    
    layers.extend([
        DQN.model.fc1.weight.T.tolist(), # weights
        DQN.model.fc1.bias.tolist(),
        DQN.model.fc2.weight.T.tolist(), # weights
        DQN.model.fc2.bias.tolist(),
        DQN.model.fc3.weight.T.tolist(), # weights
        DQN.model.fc3.bias.tolist(),
        DQN.model.fc4.weight.T.tolist(), # weights
        DQN.model.fc4.bias.tolist(),
        DQN.model.fc5.weight.T.tolist(), # weights
        DQN.model.fc5.bias.tolist(), # bias
        ])
    # Convert all layers into usable form before integrating to final agent
    layers = list(map(
        lambda x: str(list(np.round(x, 7))) \
            .replace('array(', '').replace(')', '') \
            .replace(' ', '') \
            .replace('\n', ''),
        layers
    ))
    layers = np.reshape(layers, (-1, 2))

    # Create the agent
    my_agent = '''def my_agent(observation, configuration):
    import numpy as np

    

    '''

    # Write hidden layers
    for i, (w, b) in enumerate(layers[:]):
        my_agent += 'hl{}_w = np.array({}, dtype=np.float32)\n'.format(i+1, w)
        my_agent += '    '
        my_agent += 'hl{}_b = np.array({}, dtype=np.float32)\n'.format(i+1, b)
        my_agent += '    '
   
    my_agent += '''

    state = observation.board[:]
    state.append(observation.mark)
    out = np.array(state, dtype=np.float32)
    '''

    # Calculate hidden layers
    for i in range(len(layers[:-1])):
        my_agent += '    out = np.matmul(out, hl{0}_w) + hl{0}_b\n'.format(i+1)
        my_agent += '    out = 1/(1 + np.exp(-out))\n' # Sigmoid function
    # Calculate output layer
    my_agent += '    out = np.matmul(out, hl{0}_w) + hl{0}_b\n'.format(i+2)
    my_agent += '''
    for i in range(configuration.columns):
        if observation.board[i] != 0:
            out[i] = -1e7

    return int(np.argmax(out))
    '''
    with open('submission.py', 'w') as f:
        f.write(my_agent)

def playversus(model):
    env = make('connectx', debug  = False)
    done = False
    observations, reward = env.reset()
   
    done = observations['info']
    
    observations = observations['observation']
    
    
    while not env.done:
        
        action = model.get_action(observations, 0.01)
        print(action)
        observations, reward = env.step([action, 0])    
        env.render()
        user_action = input("Enter ur move:" )
        user_action = int(user_action)
        observations, reward = env.step([0, user_action])
        observations = observations['observation']
    
    print("Done")
    env.render()

def CreateFlippedAgent(DQN):
    layers = []
    # Get layers' weights
    
    layers.extend([
        DQN.model.fc1.weight.T.tolist(), # weights
        DQN.model.fc1.bias.tolist(),
        DQN.model.fc2.weight.T.tolist(), # weights
        DQN.model.fc2.bias.tolist(),
        DQN.model.fc3.weight.T.tolist(), # weights
        DQN.model.fc3.bias.tolist(),
        DQN.model.fc4.weight.T.tolist(), # weights
        DQN.model.fc4.bias.tolist(),
        DQN.model.fc5.weight.T.tolist(), # weights
        DQN.model.fc5.bias.tolist(), # bias
        ])
    # Convert all layers into usable form before integrating to final agent
    layers = list(map(
        lambda x: str(list(np.round(x, 7))) \
            .replace('array(', '').replace(')', '') \
            .replace(' ', '') \
            .replace('\n', ''),
        layers
    ))
    layers = np.reshape(layers, (-1, 2))

    # Create the agent
    my_agent = '''
import numpy as np
'''

    # Write hidden layers
    for i, (w, b) in enumerate(layers[:]):
        my_agent += 'hl{}_w = np.array({}, dtype=np.float32)\n'.format(i+1, w)
        my_agent += 'hl{}_b = np.array({}, dtype=np.float32)\n'.format(i+1, b)
   
    
    my_agent+= '''
def predict(board): 
    out = np.matmul(board, hl1_w) + hl1_b
    out = 1/(1 + np.exp(-out))
    out = np.matmul(out, hl2_w) + hl2_b
    out = 1/(1 + np.exp(-out))
    out = np.matmul(out, hl3_w) + hl3_b
    out = 1/(1 + np.exp(-out))
    out = np.matmul(out, hl4_w) + hl4_b
    out = 1/(1 + np.exp(-out))
    out = np.matmul(out, hl5_w) + hl5_b
    return out

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
        mark = self.board[position]
        mark = int(mark)
        reward = [0, 0]
        done = False

        j = 0
        for i in range(self.num_columns):
            if not self.board[i] == 0:
                j +=1
                if j == self.num_columns:
                    done = True
                    reward = [0, 0]
                    return done, reward

        column = position % self.num_columns 
        row = int((position - column) / self.num_columns)
        
        inverse_row = self.num_rows-1-row
        
        diagonal = column + row
        inverse_diagonal = inverse_row + column
        
       
        if diagonal < self.num_rows:
            diagonal_bottom = diagonal * self.num_columns 
        else:
            diagonal_bottom =  self.num_columns * (self.num_rows-1) + (diagonal- self.num_rows) + 1 
        
        if inverse_diagonal < self.num_rows:
            inverse_diagonal_row = inverse_diagonal
            inverse_diagonal_column = 0
            diagonal_top = self.num_columns * (self.num_rows - inverse_diagonal_row-1)
        else:
            
            inverse_diagonal_column = inverse_diagonal - self.num_rows + 1
            inverse_diagonal_row = self.num_rows - 1
            diagonal_top = inverse_diagonal - self.num_rows + 1

        diagonal_column = diagonal_bottom % self.num_columns 
        diagonal_row = int((diagonal_bottom - diagonal_column) / self.num_columns)


        
        position_diagonal_ur = diagonal_bottom 
        
        position_diagonal_dr = diagonal_top

        
        position_right = row * self.num_columns
        position_vertical = column

        
        range_ur = min(self.num_columns-diagonal_column, diagonal_row+1)
        range_dr = min(self.num_columns-inverse_diagonal_column, inverse_diagonal_row+1)
        

       
        range_horizontal = self.num_columns
        range_vertical = self.num_rows

        
        win_condition_dr = 0
        win_condition_ur = 0
        win_condition_r = 0
        win_condition_down = 0

        for i in range(max(range_ur, range_dr, range_horizontal, range_vertical)):
            if range_ur >= i+1 and range_ur >= self.connect:
                if self.board[position_diagonal_ur] == mark:
                    win_condition_ur +=1
                else:
                    win_condition_ur = 0
                if win_condition_ur == self.connect:
                    reward[mark - 1] = 20 
                    reward[mark % 2 + 1 - 1] = -20
                  
                    done = True
                    return done, reward
                
                position_diagonal_ur = position_diagonal_ur - 1 * self.num_columns + 1
                
            if range_dr >= i+1 and range_dr >= self.connect:
                if self.board[position_diagonal_dr] == mark:
                    win_condition_dr +=1
                    
                else:
                    win_condition_dr = 0
                if win_condition_dr == self.connect:
                    reward[mark - 1] = 20 
                    reward[mark % 2 + 1 - 1] = -20 
                    
                    done = True
                    return done, reward
                
                
                position_diagonal_dr = position_diagonal_dr + 1 * self.num_columns + 1
                
                
            if range_horizontal >= i+1 and range_horizontal >= self.connect:
                if self.board[position_right] == mark:
                    win_condition_r +=1
                else:
                    win_condition_r = 0
                if win_condition_r == self.connect:
                    reward[mark - 1] = 20 
                    reward[mark % 2 + 1 - 1] = -20 
                    
                    done = True
                    return done, reward
                position_right = position_right + 1
                
            if range_vertical >= i+1 and range_vertical >= self.connect:
                if self.board[position_vertical] == mark:
                    win_condition_down +=1
                else:
                    win_condition_down = 0
                if win_condition_down == self.connect:
                    reward[mark - 1] = 20 
                    reward[mark % 2 + 1 - 1] = -20 
                    
                    done = True
                    return done, reward
                position_vertical = position_vertical + 1  * self.num_columns
                
                
        return done, reward


    def step(self, action, mark):
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
                observations = np.array(self.board)
                
                return observations, valid, done, reward
            else:
                print("action is full")
                valid = False
                observations = np.array(self.board)
                
                print(action)
                return observations, valid, done, reward
        else:
            print("action : ", action, end = '')
            print("is out of bounds")
            valid = False
            observations = np.array(self.board)
            return observations, valid, done, reward

    def copy_board(self, board):
        self.board = np.array(board)
    
def lookahead(EVALenv, state, action, mark, alpha, beta, depth = 2):
    num_actions = 7
    
    EVALenv.copy_board(state)
    
    new_state, valid, done, reward = EVALenv.step(action, mark)
    mark = mark%2 + 1
    if done:
        value = reward[mark-1] 
        return value 
    elif depth == 0:
        prediction = predict(np.atleast_2d(new_state))
        value = np.max(prediction)
        if value > 20:
                value = 19.999
        elif value < -20:
                value = -19.99
        return value
    else:
        value = -1e7
        possible_actions = [i for i in range(num_actions) if new_state[i] == 0]
        for action in possible_actions:
            value = max(value, -lookahead(EVALenv, new_state, action, mark, -beta, -alpha, depth -1))
            alpha = max(alpha, value)
            if alpha >= beta:
                
                break
    return value

def get_action(state):
    mark = 1
    EVALenv = ConnectXEnvironment(7, 6, 4)
    num_actions = 7
    best_value = -1e7
    best_action = 20 
    mark = int(mark)
    alpha = float("-inf")
    beta = float("inf")
    possible_actions = [i for i in range(num_actions) if state[i] == 0]
    for action in possible_actions:
        value = -lookahead(EVALenv, state, action, mark, alpha, beta)
        if value > best_value:
            best_action = action
            best_value = value
    return best_action

def my_agent(observation, configuration):
    state = observation.board[:]
    mark = observation.mark
    if mark == 2:
        flipped_board = np.array(state)
        for i in range(len(state)):
            if flipped_board[i] == 1:
                flipped_board[i] = 2
            elif flipped_board[i] == 2:
                flipped_board[i] = 1
        state = flipped_board
    out = np.array(state, dtype=np.float32)

    return get_action(out)



    '''
    with open('submission.py', 'w') as f:
        f.write(my_agent)


def playversusFlipped(model):
    env = DQN2.ConnectXEnvironment(7,6,4)
    
    done = False
    mark, observations = env.reset()
    
    env.render()
    action = input("Gjør ditt trekk:")
            
    action = int(action)
    observations, valid, done, reward  = env.step(action, mark)
    mark = 1

    while not done:

        
        if not done:  
            
            env.render()
            observations = env.flip()
            action = input("Gjør ditt trekk:")
            
            action = int(action)
            
            observations, valid, done, reward  = env.step(action, mark)
            mark = mark % 2 +1
    print("Done")
    env.render()
  


 