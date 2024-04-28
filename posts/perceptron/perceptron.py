import torch

torch.manual_seed(1234)

class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # your computation here: compute the vector of scores s
        return torch.matmul(X, torch.t(self.w))

    def predict(self, X):
        t = 0
        return 1.0*(self.score(X) > t) # shouldn't there be a defined threshold here??

class Perceptron(LinearModel):

    def loss(self, X, y):
        y_ = 2*y - 1
        return (1.0*(self.score(X)*y_ < 0)).mean()

    def grad(self, X, y):
        s_i = torch.inner(torch.t(self.w), X)
        return ((1 * (s_i*y < 0))*y*X).flatten()

    def grad_k(self, X, y, alpha, k):
        # adapted from the perceptron lecture notes
        s_i = torch.inner(torch.t(self.w), X)
        return alpha/k * torch.sum(((1 * (s_i*y < 0))[:, None]*y[:, None]*X), axis=0)

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 

    def step(self, X, y):
        self.model.loss(X, y)
        self.model.w += self.model.grad(X, y)
    
    def step_k(self, X, y, alpha, k):
        self.model.loss(X, y)
        self.model.w += self.model.grad_k(X, y, alpha, k)