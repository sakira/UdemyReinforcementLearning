# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:52:28 2019

@author: sakirahas
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta

def posterior():
    num_ones = (x == 1).sum()
    num_zeros = (x == 0).sum()
    ptheta = theta**(a + num_ones - 1) * (1 - theta)**(b + num_zeros - 1)
    return ptheta





def plot(a, b, trial, ctr):
    x = np.linspace(0, 1, 1000)
    y = beta.pdf(x, a, b)
    mean = float(a) /(a + b)
    
    plt.plot(x, y)
    plt.title('Distribution after {} trials, true rate = {}, mean= {}'.format(trial, ctr, mean))
    plt.show()
    
    
true_ctr = 0.3
a, b = 1, 1
show = [0, 5, 10, 25, 50, 100, 200, 300, 500, 700, 1000, 1500]

for t in range(1501):
    coin_toss_result = (np.random.rand() < true_ctr)
    
    if coin_toss_result:
        a += 1
    else:
        b += 1
    if t in show:
        plot(a, b, t+1, true_ctr)

