import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        x=np.array(x)
        W1=np.array(W1)
        W2 = np.array(W2)
        b1=np.array(b1)
        b2=np.array(b2)
        y_true=np.array(y_true)
        predictions = np.maximum(0,np.dot(np.maximum(0,(np.dot(x,W1.T)+b1)),W2.T)+b2)
        loss = np.mean((predictions - y_true)**2)
        dL_dz2 = (2*(predictions-y_true))/len(predictions)
        z1 = np.maximum(0,(np.dot(x,W1.T)+b1))
        dL_dz1 = np.dot(dL_dz2,W2)*(z1>0)
        # Return dict with keys:
        return{
            'loss':  np.round(np.mean((predictions-y_true)**2), 4),
            'dW1':   np.round(np.outer(dL_dz1.T,x), 4),
            'db1':   np.round(dL_dz1,4),
            'dW2':   np.round(np.outer(dL_dz2.T,np.maximum(0,(np.dot(x,W1.T)+b1))), 4),
            'db2':   np.round(dL_dz2,4)
        }
