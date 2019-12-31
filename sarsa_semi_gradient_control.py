# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:53:56 2019

@author: sakirahas
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_control import max_dict
from sarsa import random_action, SMALL_ENOUGH, GAMMA, ALPHA, ALL_POSSIBLE_ACTIONS

SA2IDX = {}
IDX = 0


class Model:
    def __init__(self):
        self.num_features = 25
        self.theta = np.random.rand(self.num_features)/np.sqrt(self.num_features)
        
    def phi(self, s, a):
        return np.array([
                s[0] - 1               if a == 'U' else 0,
                s[1] - 1.5             if a == 'U' else 0,
                (s[0]*s[1] - 3)/3      if a == 'U' else 0,
                (s[0]*s[0] - 2)/2      if a == 'U' else 0,
                (s[1]*s[1] - 4.5)/4.5  if a == 'U' else 0,
                1                      if a == 'U' else 0,
                s[0] - 1               if a == 'D' else 0,
                s[1] - 1.5             if a == 'D' else 0,
                (s[0]*s[1] - 3)/3      if a == 'D' else 0,
                (s[0]*s[0] - 2)/2      if a == 'D' else 0,
                (s[1]*s[1] - 4.5)/4.5  if a == 'D' else 0,
                1                      if a == 'D' else 0,
                s[0] - 1               if a == 'L' else 0,
                s[1] - 1.5             if a == 'L' else 0,
                (s[0]*s[1] - 3)/3      if a == 'L' else 0,
                (s[0]*s[0] - 2)/2      if a == 'L' else 0,
                (s[1]*s[1] - 4.5)/4.5  if a == 'L' else 0,
                1                      if a == 'L' else 0,
                s[0] - 1               if a == 'R' else 0,
                s[1] - 1.5             if a == 'R' else 0,
                (s[0]*s[1] - 3)/3      if a == 'R' else 0,
                (s[0]*s[0] - 2)/2      if a == 'R' else 0,
                (s[1]*s[1] - 4.5)/4.5  if a == 'R' else 0,
                1                      if a == 'R' else 0,
                1
                ])
    
    def predict(self, s, a):
        x = self.phi(s, a)
        return self.theta.dot(x)
    
    def grad(self, s, a):
        return self.phi(s, a)
    
    
def getQs(model, s):
    '''Generate the dictionary'''
    
    Qs = {}
    
    for a in ALL_POSSIBLE_ACTIONS:
        q_sa = model.predict(s, a)
        Qs[a] = q_sa
    return Qs



if __name__ == '__main__':
    
    grid = negative_grid()
    
    print('rewards')
    print_values(grid.rewards, grid)
    
        
    # Initialize SA2IDX/Q
    states = grid.all_states()
    
    for s in states:
        SA2IDX[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            SA2IDX[s][a] = IDX
            IDX += 1
            
    # initialize model
    model = Model()
            
    t = 1.0
    t2 = 1.0
    deltas = []
    
         
        
    for it in range(20000):
        if it % 100 == 0:
            t += 10e-3
            t2 += 0.01
        if it % 2000 == 0:
            print('it:', it)
            
        alpha = ALPHA / t2
        
        start_state = (2, 0)
        grid.set_state(start_state)
        
        s = start_state
        Qs = getQs(model, s)
        
        
        a, _ = max_dict(Qs)
        a = random_action(a, eps=0.5/t) # epsilon-greedy
        biggest_change = 0
        
        while not grid.game_over():
            r = grid.move(a)
            sp = grid.current_state()
            
            
                        
            old_theta = model.theta.copy()
#            print(sp, ap)
#            print(oldQsa)
#            print(alpha)
#            print(GAMMA)
#            print(Q[sp][ap])
            
            if grid.is_terminal(sp):
                G = r
            else:
                Qsp = getQs(model, sp)
                ap, _ = max_dict(Qsp)
                ap = random_action(ap, eps=0.5/t)
                G = r + GAMMA * model.predict(sp, ap)
            
            model.theta = model.theta + alpha * (G - model.predict(s,a))*model.grad(s, a)
            
            
            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())
            
            
            
            # update the state and action
            s = sp
            a = ap
            
        deltas.append(biggest_change)
        
    
    plt.plot(deltas)
    plt.show()
    
    policy = {}
    V = {}
    Q = {}
    for s in grid.actions.keys():
        Qs = getQs(model, s)
        Q[s] = Qs
        a, max_q = max_dict(Qs)
        policy[s] = a
        V[s] = max_q

    
    print('values')
    print_values(V, grid)
    print('policy')
    print_policy(policy, grid)
            