import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Step 1: Subtract max(z) for numerical stability
        stable_z = z - np.max(z)
        
        # Step 2: Compute the exponent of each element
        exp_z = np.exp(stable_z)
        
        # Step 3: Divide each exp by the sum of all exps
        softmax_probs = exp_z / np.sum(exp_z)
        
        # Step 4: Round to 4 decimal places as requested
        return np.round(softmax_probs, 4)