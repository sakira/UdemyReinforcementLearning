# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:57:13 2019

@author: sakirahas
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

SMALL_ENOUGH = 10e-4 # Threshold for convergence

def print_values(V, g):
    for i in range(g.rows):
        print('------------------------------')
        for j in range(g.cols):
            v = V.get((i,j), 0)
            if v >= 0:
                print(" %.2f|"%v, end='')
            else:
                print("%.2f|"%v, end='')
        print('')

def print_policy(P, g):
    '''Deterministic policy'''
    for i in range(g.rows):
        print('------------------------------')
        for j in range(g.cols):
            a = P.get((i,j), ' ')
            print(" %s |"%a, end='')
            
        print('')
        
        
if __name__ == '__main__':
    # iterative policy evaluation
    # given a policy, find its Value function V(s)
    
    grid = standard_grid()
    
    states = grid.all_states()
    
    ### uniformly random actions ###
    V = {}
    
    # all V(s) to 0
    for s in states:
        V[s] = 0
    gamma = 1.0 # discount factor
    
    
    while True:
        biggest_change = 0
        # loop through all the states
        for s in states:
            old_v = V[s]
            
            
            # if V[s] is not a terminal state
            if s in grid.actions:
                new_v = 0
                p_a = 1.0 / len(grid.actions[s])
                
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
                
        if biggest_change < SMALL_ENOUGH:
            break
    print('Values for uniformly random actions:')
    print_values(V, grid)
    print()
    print()
    
    ## Fixed policy
    ### fixed policy ###
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    
    
    print_policy(policy, grid)
    
    # reinitialize V[s] to zero
    V = {}
    
    for s in states:
        V[s] = 0
        
    # discout factor
    gamma = 0.9
    
    
    # 
    while True:
        biggest_change = 0
        for s in states:
            old_V = V[s]
            
            if s in policy:
                a = policy[s] # probability of that action is 1
                grid.set_state(s)
                r = grid.move(a)
                V[s] = r + gamma * V[grid.current_state()]
                biggest_change = max(biggest_change, np.abs(old_V - V[s]))
                
        if biggest_change < SMALL_ENOUGH:
            break
    print('Values for fixed policy')
    print_values(V, grid)
    
        
    