import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    # X - training samples
    # y - lables for them
    # will involve the training step and the gradient descent
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # gradient descent iterations below
        for _ in range(self.n_iters):
            # finding the approximation
            y_prediction = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_prediction - y))
            db = (1/n_samples) * np.sum(y_prediction - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    # when it gets new samples, then it can approximate the value, and return
    def predict(self, X):
        y_prediction = np.dot(X, self.weights) + self.bias
        return y_prediction