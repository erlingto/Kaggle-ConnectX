import environment
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
            env.render()
            if reward == 1:
                reward = 30
            elif reward == 0:
                reward = -30
            else :
                reward = 13

        exp = {'prev_obs': prev_observations, 'a' : action, 'r': reward, 'obs': observations, 'done' : done }
        TrainNet.add_experience(exp)

        TrainNet.train(TargetNet)
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards

def dojo(games, env, TrainNet, TargetNet, min_epsilon, epsilon, copy_step):
    decay = 0.9999
    for i in range(games):
        epsilon = max(min_epsilon, epsilon*decay)
        total_reward = generate_data(env, TrainNet, TargetNet, epsilon, copy_step)

def Agent(Model, observation):
    return Model.get_action(observation)

def playversus(model):
    env = make('connectx', debug  = False)
    done = False
    observations, reward = env.reset()
   
    done = observations['info']
    
    observations = observations['observation']
    
    
    while not env.done:
        
        action = model.get_action(observations, 0.00001)
        print(action)
        observations, reward = env.step([action, 0])    
        env.render()
        user_action = input("Enter ur move:" )
        user_action = int(user_action)
        observations, reward = env.step([0, user_action])
        observations = observations['observation']
    
    print("Done")
    env.render()