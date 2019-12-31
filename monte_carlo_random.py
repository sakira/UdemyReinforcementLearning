# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 10:11:51 2019

@author: sakirahas
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# This is only policy evaluation

def random_action(a):
    
    p = np.random.random()
    
    if p < 0.5:
        return a
    else:
        tmp = list(ALL_POSSIBLE_ACTIONS)
        tmp.remove(a)
        return np.random.choice(tmp)
    
def play_game(grid, policy):
    '''
    returns a list of states and corresponding returns
    
    '''
    
    start_states = list(grid.actions.keys())
    start_index = np.random.choice(len(start_states))
    grid.set_state(start_states[start_index])
    
    s = grid.current_state()
    
    rewards_and_states = [(s, 0)]
    
    
    # play the game
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        
        r = grid.move(a)
        s= grid.current_state()
        rewards_and_states.append((s, r))
        
    G = 0
    states_and_returns = []
    
    first = True
    
    for s, r in reversed(rewards_and_states):
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        
        G = r + GAMMA * G
    states_and_returns.reverse()
    
    return states_and_returns




if __name__ == '__main__':
    
    grid = standard_grid()
    
    print('rewards:')
    print_values(grid.rewards, grid)
    
# =============================================================================
#     
#     values
# ------------------------------
#  0.43| 0.56| 0.72| 0.00|
# ------------------------------
#  0.33| 0.00| 0.21| 0.00|
# ------------------------------
#  0.25| 0.18| 0.11|-0.17|
# =============================================================================
    
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
    
    
    for t in range(5000):
        
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
        
        

