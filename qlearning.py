# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 09:11:01 2019

@author: sakirahas
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_control import max_dict
from td_0 import random_action

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


if __name__ == '__main__':
    
    grid = negative_grid(step_cost=-0.1)
    #grid = standard_grid()
    print('rewards')
    print_values(grid.rewards, grid)
    
        
    # Initialize Q
    Q = {}
    states = grid.all_states()
    
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
            
    # update count
    update_counts = {}
    update_counts_sa = {}
    
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0
            
    t = 1.0
    deltas = []
    
         
        
    for it in range(10000):
        if it % 100 == 0:
            t += 1e-2
        if it % 2000 == 0:
            print('it:', it)
        
        start_state = (2, 0)
        grid.set_state(start_state)
        
        s = start_state
        a, _ = max_dict(Q[s])
        #a = random_action(a, eps=0.5/t) # epsilon-greedy
        biggest_change = 0
        
        while not grid.game_over():
            
            a = random_action(a, eps=0.5/t)
            r = grid.move(a)
            sp = grid.current_state()
            
            ap, max_qsap = max_dict(Q[sp])
            #ap = random_action(ap, eps=0.5/t)
            
            # update the hyperparameter alpha
            alpha = ALPHA/update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005
            
            oldQsa = Q[s][a]
#            print(sp, ap)
#            print(oldQsa)
#            print(alpha)
#            print(GAMMA)
#            print(Q[sp][ap])
            #ap, max_qsap = max_dict(Q[sp])
            
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA * max_qsap - Q[s][a])
            biggest_change = max(biggest_change, np.abs(oldQsa - Q[s][a]))
            
            update_counts[s] = update_counts.get(s, 0) + 1
            
            # update the state and action
            s = sp
            a = ap
            
        deltas.append(biggest_change)
        
    
    plt.plot(deltas)
    plt.show()
    
    policy = {}
    V = {}
    
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q
        
    print('update counts')
    print(update_counts)
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        print(v, total)
        update_counts[k] = float(v)/total
    print_values(update_counts, grid)
    
    print('values')
    print_values(V, grid)
    print('policy')
    print_policy(policy, grid)
            