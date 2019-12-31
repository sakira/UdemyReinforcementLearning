# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:52:44 2019

@author: sakirahas

Step 1: Randomly Initialize V(s) and policy pi(s)
Step 2: V(s) = iterative_policy_evaluation(pi) to find the current value of V[s]

Step 3: policy improvement
We have a flag
policy_changed = False
we loop through all the states
for s in all_states:
    
    we store old policy
    old_a = policy(s)
    policy(s) = argmax[a]{ sum[s'|r]{ p(s', a| r, a)[r + gamma * V[s']]}}
    
    if policy(s) != old_a : policy_changed = True
    if policy_changed: go back to step 2
 
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# this is deterministic
# all p(s', r|s, a) = 1 or 0

if __name__ == '__main__':
    
    # this grid gives you a reward of -0.1 for 
    # every non-terminal state
    
    grid = negative_grid(step_cost=-0.8)
    #grid = standard_grid()
    
    print('rewards: ')
    print_values(grid.rewards, grid)
    
    
    # state -> action deterministic policy
    # we randomly choose an action and updates as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    # initial policy
    print('Initial policy:')
    print_policy(policy, grid)
        
    
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else: # terminal state
            V[s] = 0
#            
#            
#    for s in states:
#        V[s] = 0
#    
    
while True:
    
    # policy evaluation step 
    while True:
        biggest_change = 0
        for s in states:
            old_V = V[s]
            
            if s in policy:
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a)
                
                V[s] = r + GAMMA * V[grid.current_state()]
                biggest_change = max(biggest_change, np.abs(old_V - V[s]))
        if biggest_change < SMALL_ENOUGH:
            break
    
    
    # policy improvement step

    is_policy_converged = True
    for s in states:
        if s in policy:
            # grab that action
            old_a = policy[s]
            new_a = None
            
            best_value = float('-inf')
            
            # loop through all possible actions to find
            # the best current action
            
            for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                v = r + GAMMA * V[grid.current_state()]
                
                if v > best_value:
                    best_value = v
                    new_a = a
            # we update our policy
            
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False
    
    if is_policy_converged:
        break
    
    print('values')
    print_values(V, grid)
    print('policy')
    print_policy(policy, grid)
    
    
    
