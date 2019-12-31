# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:45:44 2019

@author: sakirahas
"""

# =============================================================================
# # Psuedocode
# for t in range(max_iterations):
#     state_history = play_game
#     for (s, s') in state_history from end to start:
#         V(s) = V(s) + learning_rate * (V(s') - V(s))
# =============================================================================

# =============================================================================
# # We loop through all the possible actions and next states
# # and check V(s')
# 
# maxV = 0 
# maxA = None
# 
# for a, s' in possible_next_states:
#     if V(s') > maxV:
#         maxV = V(s')
#         maxA = a
# perform action maxA
# 
# =============================================================================

LENGTH = 3

import numpy as np
from builtins import input

class Human():
    def __init__(self):
        pass
    
    def set_symbol(self, symbol):
        self.symbol = symbol
        
    def take_action(self, env):
        while True:
            move = input('Enter cordinates i, j for your next move(): ')
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env.board[i, j] = self.symbol
                break
    def update(self, env):
        pass
    
    def update_state_history(self, s):
        pass
            
class Agent():
    
    def __init__(self, name, eps=0.1, alpha=0.5):
        self.name = name
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []
        
    def setV(self, V):
        self.V = V
        
    def set_symbol(self, symbol):
        self.symbol = symbol
        
    def set_verbose(self, b):
        self.verbose = b
        
        
    def reset_history(self):
        self.state_history = []
    
    def take_action(self, env):
        '''Choose an action based on epsilon-greedy strategy'''
        
        r = np.random.rand()
        
        best_state = None
        if r < self.eps:
            # take a random action
            if self.verbose:
                print('Taking a random action')
                
            possible_moves = []
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        possible_moves.append((i, j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
            next_move = None
            best_value = -1
            
            
            pos2value = {} # for debugging
            
            
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        env.board[i, j] = self.symbol
                        state = env.get_state()
                        env.board[i, j] = 0
                        
                        pos2value[(i,j)] = self.V[state]
                        
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i, j)
            
            if self.verbose:
                print('Taking a greedy action')
                for i in range(LENGTH):
                    
                    print('---------------------')
                    for j in range(LENGTH):
                        if env.is_empty(i, j):
                            print('%.2f|'% pos2value[(i,j)], end='')
                        else:
                            print(' ', end='')
                            if env.board[i, j] == env.x:
                                print('x|', end='')
                            elif env.board[i, j] == env.o:
                                print('o|', end='')
                            else:
                                print(' |', end='')
                    print('')
                print('---------------------------')
        # make the move
        env.board[next_move[0], next_move[1]] = self.symbol
            
        
    def update_state_history(self, s):
        self.state_history.append(s)
    
    def update(self, env):
        reward = env.reward(self.symbol)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()
        
    
        
class Env():
    def __init__(self):
        self.x  = -1
        self.o = 1
        self.board = np.zeros((LENGTH, LENGTH))
        self.winner = None
        self.ended = False
        self.num_states = 3**(LENGTH * LENGTH)

    def is_empty(self, i, j):
        ''' If a location i, j is empty or not'''
        return (self.board[i, j] == 0)

    def reward(self, symbol):
        if not self.game_over():
            return 0
        return 1 if self.winner == symbol else 0
    
    def draw_board(self):
        '''Draw Tic-Tac-Toe board'''
        for i in range(LENGTH):
            print("---------------------")
            for j in range(LENGTH):
                print(' ', end='')
                if self.board[i, j] == self.x:
                    print('x', end='')
                elif self.board[i, j] == self.o:
                    print('o', end='')
                else:
                    print(' ',end='')
            print('')
        print('---------------------')
        
        
    def game_over(self, force_recalculate=False):
        '''
        Scan the board and check if there is any winner or draw
        '''
        if not force_recalculate and self.ended:
            return self.ended
        
        # check rows
        for i in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[i].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = True
                    return True
        # check columns
        for j in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[:, j].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = True
                    return True
        
        
        # check diagonals
        for player in (self.x, self.o):
            # top-left -> bottom-right
            if self.board.trace() == player*LENGTH:
                self.winner = player
                self.ended = True
                return True
            # top-right --> bottom-left
            if np.fliplr(self.board).trace() == player*LENGTH:
                self.winner = player
                self.ended = True
                return True
            
        
        # check if draw
        if np.all((self.board == 0) == False):
            self.winner = None
            self.ended = True
            return True
        
        self.winner = None
        return False
        
        

    def get_state(self):
        h = 0
        k = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (3**k) * v
                k += 1
        return h

# initialize Vx and Vo
def initialV_x(env, state_winner_triples):
    # initialize Vx
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V

def initialV_o(env, state_winner_triples):
    # initialize Vx
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V

def get_state_hash_and_winner(env, i=0, j=0):
    '''
    Enumerating states recursively
    '''
    #state = configure_board()
    results = []
    
    
    for v in (0, env.x, env.o):
        env.board[i, j] = v # if empty board, it should already be 0
        
        if j == 2:
            # j goes back to 0, increase i
            if i == 2: # unless i = 2, we are done
                # the board is full, collect results and return
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i+1, 0)
        else:
            results += get_state_hash_and_winner(env, i, j+1)
    
    
    return results

def generate_all_binary_numbers(N):
    '''How to generate a recursive function'''
    results = []
    
    if N == 0:
        return ('0', '1')
    
    child_results = generate_all_binary_numbers(N-1)
    for prefix in ('0', '1'):
        for suffix in child_results:
            new_result = prefix + suffix
            results.append(new_result)
    return results
    
def play_game(p1, p2, env, draw=False):
    '''
    Play game: two instances of Agent
    interacting with the same environment object.
    '''
    
    current_player = None
    
    while not env.game_over():
    
        # switch between the player
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        
        # draw the board before the user who wants to see it makes a move
        if draw:
            if draw == 1 and current_player == p1:
                env.draw_board()
            if draw == 2 and current_player == p2:
                env.draw_board()
        
        
        # current player take action which update the environment
        current_player.take_action(env)
        
        
        # update state history
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
        
    if draw:
        env.draw_board()
    
    # estimate the value function
    p1.update(env)
    p2.update(env)
    
    
    

if __name__ == '__main__':
    
    # Initialize two players
    p1 = Agent('p1')
    p2 = Agent('p2')
    
    env = Env()
    state_winner_triples = get_state_hash_and_winner(env)
    
    Vx = initialV_x(env, state_winner_triples)
    Vo = initialV_o(env, state_winner_triples)
    
    p1.setV(Vx)
    p2.setV(Vo)
    
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)
    
    
    # Training
    
    max_iterations = 10000
    
    
    for t in range(max_iterations):
        if t % 200 == 0:
            print(t)
        play_game(p1, p2, Env()) # create new environment at every iteration
        
    
    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, Env(), draw=2)
        
        answer = input('Play again?[Y/n]: ')
        if answer and answer.lower()[0] == 'n':
            break
    