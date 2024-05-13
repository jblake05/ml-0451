import torch
import numpy as np

class LinearModel:

    def __init__(self):
        """
        Initialize the LinearModel with a default weight

        ARGUMENTS:
            None
        
        RETURNS:
            None
        """
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

        # initializes model's weight vector if not already done
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))/1000

        # compute the vector of scores s
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

        # defines a threshold at 0, returns 1 if the score is higher than it, 0 if not
        t = 0
        return 1.0*(self.score(X) > t)

class LogisticRegression(LinearModel):

    def sigmoid(self, s):
        """
        Compute the sigmoid function for a vector of scores s

        ARGUMENTS:
            s, torch.Tensor: the score vector computed using self.score. In practice 
            (as implemented in the loss and grad functions), s.size() == (n,), where
            n is the number of data points.
        
        RETURNS:
            σ(s), torch.Tensor: a vector of the scores put through the sigmoid function. 
            σ(s).size() == s.size() == (n,), where n is the number of data points.
        """
        return 1/(1 + np.exp(-s))

    def loss(self, X, y):
        """
        Compute the loss, comparing y to the scores the model generated

        ARGUMENTS:
            X, torch.Tensor (description used from above): the feature matrix.
            X.size() == (n, p), where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector. y.size() == (n,) where n is the number of
            data points.
        
        RETURNS:
            L(w), torch.Tensor: The loss of model on target vector y. Given the weights of the
            model. this function determines a level of correctness, computing how well the score
            values of the model on X match with the target of y. L(w).size() == (1,).
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))/1000

        s = self.score(X)

        return torch.mean(-y*np.log(self.sigmoid(s)) - (1 - y) * np.log(1 - self.sigmoid(s)))
    
    def grad(self, X, y):
        """
        Calculate the gradient for the loss function
        
        ARGUMENTS:
            X, torch.Tensor (description used from above): the feature matrix.
            X.size() == (n, p), where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector. y.size() == (n,) where n is the number of
            data points.
        
        RETURNS:
            ΔL(w), torch.Tensor: The gradient of the loss function. ΔL(w).size() == (p,),
            where p is the number of features.
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))/1000

        s = self.score(X)

        return torch.mean((self.sigmoid(s) - y)[:, None]*X, axis=0)

class GradientDescentOptimizer:
    def __init__(self, model):
        """
        Initialize the gradient descent optimizer with a model and a default previous weight

        ARGUMENTS:
            model, LogisticRegression: The model this class optimizes weights for

        RETURNS:
            None
        """
        self.model = model
        self.prev_w = None

    def step(self, X, y, alpha, beta):
        """
        Performs a step of optimization for the logistic regression model, updating its weight.

        ARGUMENTS:
            X, torch.Tensor (description used from above): the feature matrix.
            X.size() == (n, p), where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector. y.size() == (n,) where n is the number of
            data points.

            alpha, float: Learning rate

            beta, float: Momentum value 

        RETURNS:
            None
        """

        # Intializes the model's weight and previous weight to prevent errors caused by doing math with None types 
        if self.model.w is None or self.prev_w is None:
            self.model.w = torch.rand((X.size()[1]))/1000
            self.prev_w = torch.rand((X.size()[1]))/1000

        temp = self.model.w

        # Updates the new model using the gradient loss, updates prev_w
        self.model.w = self.model.w - alpha*self.model.grad(X, y) + beta*(self.model.w - self.prev_w)
        self.prev_w = temp
