import torch
import numpy as np

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # your computation here: compute the vector of scores s
        return torch.matmul(X, torch.t(self.w))

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        t = 0
        return 1.0*(self.score(X) > t) # shouldn't there be a defined threshold here??

class LogisticRegression(LinearModel):

    # shorthand for the sigmoid function
    def sigmoid(self, s):
        return 1/(1 + np.exp(-s))

    def loss(self, X, y):
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        s = self.score(X)

        return torch.mean(-y*np.log(self.sigmoid(s)) - (1 - y) * np.log(1 - self.sigmoid(s)))
    
    def grad(self, X, y):
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        s = self.score(X)

        return torch.mean((self.sigmoid(s) - y)[:, None]*X, axis=0)

class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model
        self.prev_w = None

    def step(self, X, y, alpha, beta):
        if self.model.w is None or self.prev_w is None:
            self.model.w = torch.rand((X.size()[1]))
            self.prev_w = torch.rand((X.size()[1]))

        temp = self.model.w

        self.model.w = self.model.w - alpha*self.model.grad(X, y) + beta*(self.model.w - self.prev_w)
        self.prev_w = temp
