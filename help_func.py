import environment
import types
import torch
import network
import numpy as np
import IPython
from kaggle_environments import evaluate, make, utils


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
    if observation.mark == 2:

        flipped_board = np.array(state)
            for i in range(len(state)):
                if flipped_board[i] == 1:
                    flipped_board[i] = 2
                elif flipped_board[i] == 2:
                    flipped_board[i] = 1
        state = flipped_board
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


def playversusFlipped(model):
    env = make('connectx', debug  = False)
    done = False
    observations, reward = env.reset()
   
    done = observations['info']
    
    observations = observations['observation']
    observations = observations.board
    
    
    while not env.done:
        
        action = model.get_action(observations, 0.0)
        action = int(action)
        print(action)
        observations, reward = env.step([action, 0])    
        env.render()
        user_action = input("Enter ur move:" )
        user_action = int(user_action)
        print(user_action)
        observations, reward = env.step([0, user_action])
        observations = observations['observation']
        observations = observations.board
    
    print("Done")
    env.render()

 


 