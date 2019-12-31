# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:01:49 2019

@author: sakirahas
"""

import argparse
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


from data_model import maybe_make_dir
from data_model import get_data
from data_model import get_scaler

from environment import MultiStockEnvironment
from agent import DQNAgent

def play_one_episode(agent, env, train_mode):
    # reset the environment to get to the intial state
    state = env.reset()
    state = scaler.transform([state])
    done = False # end of the game
    
    while not done:
        # choose an action
        action = agent.act(state) # determine the next action by agent
        # perform the action to get to the next state, reward
        next_s, reward, done, info = env.step(action)
        
        next_s = scaler.transform([next_s]) # normalize
        
        if train_mode: # check the script is in train_mode
            agent.train(state, action, reward, next_s, done)
        
        # update the state
        state = next_s
        
        
    return info['cur_val']



if __name__ == '__main__':
    
    # config
    models_folder = 'linear_rl_trader_models'
    rewards_folder = 'linear_rl_trader_rewards'
    num_episodes = 2000
    batch_size = 32
    initial_investment = 20000
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either train or test')
    
    arg = parser.parse_args()
    
    # create directory
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)
    
    # fetch the data
    data = get_data()
    n_timesteps, n_stocks = data.shape
    
    n_train = n_timesteps // 2
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    
    
    
    
    # Create an instance of the environment
    env = MultiStockEnvironment(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    
    # Create an instance of the agent
    agent = DQNAgent(state_size, action_size)
    
    
    scaler = get_scaler(env)
    
    # Initialize the portfolio values
    portfolio_values = []
    
    if arg.mode == 'test':
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        env = MultiStockEnvironment(test_data, initial_investment)
        
        agent.epsilon = 0.01
        
        agent.load(f'{models_folder}/linear.npz')
        
        
    
    # loop for going through one episode
    for e in range(num_episodes):
        
        t0 = datetime.now()
        val = play_one_episode(agent, env, arg.mode)
        dt = datetime.now() - t0
        
        
        print(f'episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}')
        portfolio_values.append(val)
        
    if arg.mode == 'train':
        agent.save(f'{models_folder}/linear.npz')
        
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        plt.plot(agent.model.losses)
        plt.show()
        
    
    np.save(f'{rewards_folder}/{arg.mode}.npy', portfolio_values)