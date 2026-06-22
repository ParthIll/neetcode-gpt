import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        
        # 1. Define the embedding layer mapping vocabulary to 16 dimensions
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16)
        
        # 2. Define the linear layer mapping 16 dimensions to a single logit output
        self.linear = nn.Linear(in_features=16, out_features=1)
        
        # 3. Define the Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Input 'x' has shape: [Batch Size (B), Sequence Length (T)]
        
        # 1. Pass through embedding. Output shape: [B, T, 16]
        out = self.embedding(x)
        
        # 2. Collapse the sequence length 'T' by taking the mean across dim=1. 
        # Output shape: [B, 16]
        out = torch.mean(out, dim=1)
        
        # 3. Pass through the linear layer. Output shape: [B, 1]
        out = self.linear(out)
        
        # 4. Apply Sigmoid to get probabilities between 0 and 1. Output shape: [B, 1]
        out = self.sigmoid(out)
        
        # 5. Round to 4 decimal places as requested
        return torch.round(out, decimals=4)