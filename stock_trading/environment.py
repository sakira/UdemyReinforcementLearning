import numpy as np

import itertools

class Environment:
    def __init__(self, stock_prices, initial_investment):
        self.pointer = 0 # points to what day it is to know the current day stock price 
        self.stock_prices = stock_prices # timeseries of stock prices 
        self.initial_investment = initial_investment # How cash initially
        
    def reset(self):
        '''
        Set the pointer to zero and recalculate all the states. All cash and no investment.
        '''
        self.pointer = 0
        
    def step(self, action):
        '''
        Take action: buy stock specified by the action
        set the pointer to the next days stock price
        done will be set to true if we reach our timeseries
        '''
        pass
    
    
class MultiStockEnvironment:
    def __init__(self, data, initial_investment):
        # data
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape
        
        # instance attributes
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        
        
        self.action_space = np.arange(3**self.n_stock)
        
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        
        self.state_dim = self.n_stock * 2 + 1
        
        self.reset()
        
    def reset(self):
        
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        
        return self._get_obs()
    def step(self, action):
        assert action in self.action_space
        
        # get the current porfolio
        prev_val = self._get_val()
        
        # update price go to next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]
        
        # perform the trade
        self._trade(action)
        
        # get the new portfolio value
        cur_val = self._get_val()
        
        # calcualte the reward
        reward = cur_val - prev_val
        
        # run out of data
        done = self.cur_step == self.n_step - 1
        
        info = {'cur_val': cur_val}
        
        return self._get_obs(), reward, done, info
    
    
    def _get_obs(self):
        '''
        return the state
        '''
        obs = np.empty(self.state_dim)
        # number of stock owned
        obs[:self.n_stock] = self.stock_owned
        # value of the stock
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        # cash in hand
        obs[-1] = self.cash_in_hand
        return obs
        pass
    
    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand
        pass
    
    def _trade(self, action):
        '''
        0 = sell
        1 = hold
        2 = buy
        '''
        # grab action vector
        action_vec = self.action_list[action]
        
        sell_index = []
        buy_index = []
        
        # determine which stocks to buy or sell
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)
                
        # sell any stock, then buy
        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                    
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False
                        
                    
        
        
        
        
        
        
        
        
        
        