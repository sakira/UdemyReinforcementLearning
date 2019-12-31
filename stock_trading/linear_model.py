import numpy as np

class LinearModel:
    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action)/np.sqrt(input_dim)
        self.b = np.zeros(n_action)
        
        self.vW = 0
        self.vb = 0
        
        self.losses = []
        
    def predict(self, X):
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b
    
    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        assert(len(X.shape) == 2)
        
        num_values = np.prod(Y.shape)
        
        Yhat = self.predict(X)
        gW = 2 / num_values * X.T.dot(Yhat - Y)
        gb = 2 / num_values * (Yhat - Y).sum(axis=0)
        
        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb
        
        # update params
        self.W = self.W + self.vW
        self.b = self.b + self.vb
        
        mse = np.mean((Y - Yhat)**2)
        self.losses.append(mse)
        
    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']
        
    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)