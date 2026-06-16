import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Vectorized Sigmoid: 1 / (1 + e^(-z))
        result = 1 / (1 + np.exp(-z))
        return np.round(result, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Vectorized ReLU: takes the element-wise maximum of 0 and z
        return np.maximum(0, z)