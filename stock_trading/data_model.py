import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def get_data():
    df = pd.read_csv('data/aapl_msi_sbux.csv')
    return df.values


def get_scaler(env):
    '''Run it for multiple episodes '''
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    