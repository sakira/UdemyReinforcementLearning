# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 20:09:39 2019

@author: sakirahas

Q = random
pi = random

while True:
    s, a = randomly select from S and A
    states_actions_returns = play_game(start=(s,a))
    for s, a, G in states_actions_returns:
        returns(s, a).append(G)
        Q(s, a) = average(returns(s, a))
    for s in states:
        pi(s) = argmax[a]{Q(s, a)}

"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def play_game(grid, policy):
    
    start_states = list(grid.actions.keys())
    start_index = np.random.choice(len(start_states))
    grid.set_state(start_states[start_index])
    
    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    
    states_actions_rewards = [(s, a, 0)]
    seen_states = set()
    seen_states.add(grid.current_state())
    num_steps = 0
    
    while True:
        
        #old_s = grid.current_state()
        print(a)
        r = grid.move(a)
        num_steps += 1
        
        s = grid.current_state()
        
        #if old_s == s:
        if s in seen_states:
            reward = -10./num_steps
            states_actions_rewards.append([s, None, -100])
            #states_actions_rewards.append((s, None, reward))
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s, a, r))
        seen_states.add(s)
    # compute the return
    G = 0
    states_actions_returns = []
    
    first = True
    
    for s, a, r in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
            
        G = r + GAMMA * G
    states_actions_returns.reverse()
    
    return states_actions_returns



def max_dict(d):
    max_key = None
    max_val = float('-inf')
    
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val



    
def random_actions_states(s, a):
    
    p = np.random.random()
    
    if p < 0.5:
        return (s, a)
    else:
        tmp = list(ALL_POSSIBLE_ACTIONS)
        tmp.remove(a)
        a = np.random.choice(tmp)
        
        return (s, a)

if __name__ == '__main__':
    
    grid = standard_grid()
    grid = negative_grid(step_cost=-0.9)
    
    
    print('rewards')
    print_values(grid.rewards, grid)
    
    
    # initialize a random policy
    
    policy = {}
    
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
        
        
    
    print('policy')
    print_policy(policy, grid)
    
    Q = {}
    returns = {}
    
    states = grid.all_states()
    actions = grid.actions
    
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s, a)] = []
        else:
            pass
        
    # repeat until convergence
    deltas = []
    
    for t in range(2000):
        
        if t%1000 == 0:
            print(t)
        
        biggest_change = 0
#        
#        states_actions_returns = play_game(start=(s,a))
#    for s, a, G in states_actions_returns:
#        returns(s, a).append(G)
#        Q(s, a) = average(returns(s, a))
#    for s in states:
#        pi(s) = argmax[a]{Q(s, a)}
        
        print('At t:', t)
        print(policy)
        
        states_actions_returns = play_game(grid, policy)
        # create a set
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_Q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                seen_state_action_pairs.add(sa)
        
                biggest_change = max(biggest_change, np.abs(old_Q - Q[s][a]))
        deltas.append(biggest_change)
        
        for s in policy.keys():
            #policy[s] = np.argmax(Q[s])
            policy[s] = max_dict(Q[s])[0]
        
    
    plt.plot(deltas)
    plt.show()
    
    print('final policy')
    print_policy(policy, grid)
    


    # find V
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]
        
    print('final value')
    print_values(V, grid)