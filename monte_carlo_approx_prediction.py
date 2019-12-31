# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:16:24 2019

@author: sakirahas
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 09:42:05 2019

@author: sakirahas

def mc-approx-prediction(policy):
    theta = randomly initialize
    for i = 1...N:
        states_and_returns = play_game()
        for s, g in states_and_returns:
            x = phi(s)
            theta = theta + alpha * (g - theta.T * x) * x


"""

import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
import matplotlib.pyplot as plt

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
LEARNING_RATE = 0.001


def random_action(a):
    
    p = np.random.random()
    
    if p < 0.5:
        return a
    else:
        tmp = list(ALL_POSSIBLE_ACTIONS)
        tmp.remove(a)
        return np.random.choice(tmp)

# NOTE: this in only policy evaluation, not optimization

def play_game(grid, policy):
    '''
    returns a list of state and corresponding returns
    '''
    
    # reset the game at random position - Explore game
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])
    
    # calculating returns and rewards
    
    # First state
    s = grid.current_state()
    
    states_and_rewards = [(s, 0)]
    
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        # updated states
        s = grid.current_state()
        states_and_rewards.append((s, r))
        
    # calculate rewards in reversed order
    G = 0
    
    states_and_returns = []
    first = True
    
    for s, r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        
        G = r + GAMMA * G
        
    states_and_returns.reverse()
    
    
    return states_and_returns


if __name__ == '__main__':
    
    grid = standard_grid()
    #grid = negative_grid()
    
    print('rewards')
    print_values(grid.rewards, grid)
    
    # create the policy
    # state --> action
    policy = {
        (2, 0) : 'U',
        (1, 0) : 'U',
        (0, 0) : 'R',
        (0, 1) : 'R',
        (0, 2) : 'R',
        (1, 2) : 'U',
        (2, 1) : 'L',
        (2, 2) : 'U',
        (2, 3) : 'L',        
    }
    print('initial policy')
    print_policy(policy, grid)
    
    
    # intialize the returns
    
    V = {}
    returns = {}
    
    states = grid.all_states()
    
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0
    
    
    # monte carlo loop
    theta = np.random.rand(4) / 2
    # function transform s to feature vector x
    def phi(s):
        return np.array([
                s[0] - 1,
                s[1] - 1.5,
                s[0]*s[1] - 3,
                1
                ])
    
    
    deltas = []
    t = 1.0
    for it in range(20000):
        
        if it % 100 == 0:
            t += 0.01
        
        if t % 1000 == 0:
            print(t)
        # decaying rate
        alpha = LEARNING_RATE / t
        biggest_change = 0
        
        # generate an episode
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        
        for s, G in states_and_returns:
            if s not in seen_states:
                old_theta = theta.copy()
                x = phi(s)
                V_hat = theta.dot(x)
                theta = theta + alpha * (G - V_hat) * x
                
                
                seen_states.add(s)
                biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
        deltas.append(biggest_change)
              
    plt.plot(deltas)
    plt.show()
    
    V = {}
    for s in grid.all_states():
        if s in grid.actions:
            V[s] = theta.dot(phi(s))
        else:
            V[s] = 0
    
    
    print('values')
    print_values(V, grid)
    print('policy')
    print_policy(policy, grid)
        
        
        
        
        
        
        