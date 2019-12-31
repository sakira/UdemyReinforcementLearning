# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 09:42:05 2019

@author: sakirahas
"""

import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

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
    grid = negative_grid()
    
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
        (1, 2) : 'R',
        (2, 1) : 'R',
        (2, 2) : 'R',
        (2, 3) : 'U',        
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
    
    for t in range(100):
        
        # generate an episode
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        
        for s, G in states_and_returns:
            if s not in seen_states:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)
    print('values')
    print_values(V, grid)
    print('policy')
    print_policy(policy, grid)
        
        
        
        
        
        
        