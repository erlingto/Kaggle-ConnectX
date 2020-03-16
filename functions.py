import environment
import torch
import network
import numpy as np

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