# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:28:21 2019

@author: sakirahas
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 10:35:21 2019

@author: sakirahas
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 09:30:14 2019

@author: sakirahas
Design Bandit methods
"""


import numpy as np
import matplotlib.pyplot as plt
from epsilon_greedy import run_experiment as run_experiment_eps
# =============================================================================
# # psuedo code for Bandit experiment
# eps = 0.1
# p = np.random()
# 
# if p < eps:
#     # pull random arm
#     pass
# else:
#     # pull current best arm
#     pass
#     
# =============================================================================
# =============================================================================
# #psuedo code for supervised learning
#    
# class MyModel():
#     
#     def fit(X, Y):
#         pass
#     def predict(X):
#         
#         pass
#     
# 
# # Load the data
# Xtrain, Ytrain, Xtest, Ytest = load_data()
# # Instantiate the model
# model = MyModel()
# # Train the model
# model.fit(Xtrain, Ytrain)
# # Evaluate the model
# model.score(Xtest, Ytest)
# =============================================================================
# =============================================================================
# # psuedo code for reinforcement learning
# class MyAwesomeCasinoMachine:
#     def pull():
#         # simulate drawing from the true distribution
#         # which you wouldn't know in real life
# 
# for t in range(max_iterations):
#     # pick casino machine to play based on algorithm
#     # update algorithm parameters
#     
# # plot useful info (avg reward, best machine, etc.)
# =============================================================================

class Bandit():
    def __init__(self, m, upper_limit):
        # takes true mean m as argument
        self.m = m
        self.mean = upper_limit
        self.N = 0
    
    def pull(self):
        # simulate drawing from the true distribution
        # which you wouldn't know in real life
        return np.random.randn() + self.m
    
    def update(self, x):
        # update the mean and N
        self.N += 1
        self.mean = (1 - 1.0/self.N) * self.mean + 1.0/self.N * x
        
def run_experiment(m1, m2, m3, N, upper_limit=10):
    # takes three different Bandit m1, m2 and m3
    bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]

    data = np.empty(N)
    
    # run the experiment for N times
    for i in list(range(N)):
        # choose the current best arm
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    
    
    # plot moving average ctr
    
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()
    
    
    # print estimated means
    for b in bandits:
        print(b.mean)
    
    return cumulative_average

if __name__ == '__main__':
    
    c_1 = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
    c_05 = run_experiment(1.0, 2.0, 3.0, 100000)
    
    
    # log scale plot
    plt.plot(c_1, label='eps=0.1')
    plt.plot(c_05, label='optimistic')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    # linear plt
    plt.plot(c_1, label='eps=0.1')
    plt.plot(c_05, label='optimistic')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    