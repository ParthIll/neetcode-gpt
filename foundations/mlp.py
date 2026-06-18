import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        inputLayer = x
        weightLayer = weights[0]
        biasLayer = biases[0]
        for i in range(len(weights)-1):
            weightLayer = weights[i]
            biasLayer = biases[i]
            inputLayer = np.maximum(0,np.round(np.dot(inputLayer,weightLayer)+biasLayer,5))
        return  np.round(np.dot(inputLayer,weights[-1])+biases[-1],5)
