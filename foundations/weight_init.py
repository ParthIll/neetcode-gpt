import torch
import torch.nn as nn
import math
from typing import List
import numpy as np

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return np.round(weights.numpy(), 4).tolist()
        

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return np.round(weights.numpy(), 4).tolist()


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # 1. Seed once at the very beginning
        torch.manual_seed(0)
        
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        
        # 2. Generate all layer weights first to preserve the precise RNG order
        for i in range(num_layers):
            if init_type == "kaiming":
                std = math.sqrt(2.0 / dims[i])
            elif init_type == "xavier":
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            else:
                std = 1.0
                
            # Shape is (fan_out, fan_in)
            w = torch.randn(dims[i+1], dims[i]) * std
            weights.append(w)
            
        # 3. Generate the input tensor *after* all weights have been created
        # Note: Neetcode expects an explicit 2D batch dimension shape of (1, input_dim)
        x = torch.randn(1, input_dim)
        
        stds = []
        
        # 4. Forward pass through the pre-generated layers
        for w in weights:
            x = x @ w.t()
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))
            
        return stds