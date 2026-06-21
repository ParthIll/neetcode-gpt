import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        w=np.zeros(X.shape[1])
        b=0
        
        
        for i in range(epochs):
            y_hat = np.dot(X,w) + b
            MSE = (1/len(X)) * sum((y_hat - y)**2)
            dLdW = 2/len(X)*X.T@(y_hat-y)
            dLdb = 2/len(X)*np.sum(y_hat-y)
            w = w-(lr * dLdW)
            b = b - (lr * dLdb)
        # Initialize w = zeros, b = 0
        return (np.round(w, 5), round(b, 5))
        
