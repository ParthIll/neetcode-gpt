import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x, dtype=np.float64)
        
        # ---> CHANGE 2: Initialize y with the same shape so y[:, feature] works
        y = np.zeros_like(x)
        if(training):
            for feature in range(len(gamma)):
                mean = 1/len(x)*np.sum(x,axis=0)[feature]
                variance = 1/len(x)*np.sum((x-mean)**2,axis=0)[feature]
                running_mean[feature] = (1-momentum)*running_mean[feature] + momentum*mean
                running_var[feature]= (1-momentum)*running_var[feature] + momentum*variance
                x_hat = x[:,feature]
                x_hat = (x_hat-mean)/(np.sqrt(variance+eps))
                y[:,feature] = gamma[feature]*x_hat+beta[feature]
            
        else:
            for feature in range(len(gamma)):
                x_hat =x[:,feature]
                x_hat = (x_hat-running_mean[feature])/(np.sqrt(running_var[feature]+eps))
                y[:,feature]=gamma[feature]*x_hat+beta[feature]
        return (np.round(y,4),np.round(running_mean,4),np.round(running_var,4))

