# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:00:57 2019

@author: sakirahas
"""

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

from optimistic_initial_values import run_experiment as run_experiment_oiv
from ucb1 import run_experiment as run_experiment_ucb
from epsilon_greedy import Bandit

#from epsilon_greedy import run_experiment as run_experiment_eps
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

def run_experiment_epsilon_decay(m1, m2, m3,  N):
    # takes three different Bandit m1, m2 and m3
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)
    
    # run the experiment for N times
    for i in range(N):
        # epsilon-greedy
        p = np.random.random()
        eps = 1.0 /(i+1)
        if p < 1.0 /(i+1):
            # choose random arm
            j = np.random.choice(3)
        else:
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



class BayesianBandit():
    def __init__(self, m):
        # takes true mean m as argument
        self.m = m
        #self.mean = 0
        #self.N = 0
        self.m0 = 0
        self.lmbda0 = 1
        self.sum_x = 0
        self.tau = 1
    
    def pull(self):
        # simulate drawing from the true distribution
        # which you wouldn't know in real life
        return np.random.randn() + self.m
    
    def sample(self):
        '''
        Generate the samples from gaussian distribution param 
        lmbda0 and m0
        '''
        return np.random.randn() / np.sqrt(self.lmbda0) + self.m0
    
    def update(self, x):
        # update the mean and N
        #self.N += 1
        #self.mean = (1 - 1.0/self.N) * self.mean + 1.0/self.N * x
        self.lmbda0 += self.tau
        self.sum_x += x
        self.m0 =self.tau * self.sum_x / self.lmbda0
        #lmbda = tau * self.N + self.lmbda0
        #self.mean = (self.tau * self.sum_x + self.m0 * self.lmbda0 ) / lmbda

    
def run_experiment(m1, m2, m3, N, upper_limit=10):
    # takes three different Bandit m1, m2 and m3
    #bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]
    bandits = [BayesianBandit(m1), BayesianBandit(m2), BayesianBandit(m3)]

    data = np.empty(N)
    
    # run the experiment for N times
    for i in list(range(N)):
        # choose the current best arm
        j = np.argmax([b.sample() for b in bandits])
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
    
    
#    # print estimated means
#    for b in bandits:
#        print(b.mean)
#    
    return cumulative_average

if __name__ == '__main__':
    
    eps = run_experiment_epsilon_decay(1.0, 2.0, 3.0,  100000)
    oiv = run_experiment_oiv(1.0, 2.0, 3.0, 100000)
    ucb = run_experiment_ucb(1.0, 2.0, 3.0, 100000)
    bayes = run_experiment(1.0, 2.0, 3.0, 100000)
    
    
    # log scale plot
    plt.plot(eps, label='decaying-epsilon-greedy')
    plt.plot(oiv, label='optimistic')
    plt.plot(ucb, label='ucb1')
    plt.plot(bayes, label='bayesian')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
    # linear plt
    plt.plot(eps, label='decaying-epsilon-greedy')
    plt.plot(oiv, label='optimistic')
    plt.plot(ucb, label='ucb1')
    plt.plot(bayes, label='bayesian')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    