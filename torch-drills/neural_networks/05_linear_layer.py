"""
DRILL 5: Implementing a Linear Layer from Scratch

TASK: Implement a linear (fully connected) layer without using nn.Linear:

1. Create a LinearLayer class with __init__ and forward methods
2. Initialize weights with Xavier/Glorot initialization
3. Initialize biases to zero
4. Implement forward pass: output = input @ weight + bias
5. Handle proper tensor dimensions
6. Compare your implementation with torch.nn.Linear

EXPECTED OUTPUT:
- LinearLayer class that works like nn.Linear
- Proper weight initialization
- Forward pass should produce same results as nn.Linear (within tolerance)
"""

import torch
import torch.nn as nn
import math

class LinearLayer:
    def __init__(self, in_features, out_features):
        # YOUR CODE HERE

        self.bias = torch.zeros(out_features,)
        self.weight = torch.zeros(out_features, in_features) # weights are this way bc backwards comp
        nn.init.xavier_normal_(self.weight)
    
    def forward(self, x):
        # YOUR CODE HERE
        self.output = torch.matmul(x, self.weight.T) + self.bias
        return self.output

def solve():
    # Test your implementation
    batch_size, in_features, out_features = 32, 10, 5
    
    # Create test input
    x = torch.randn(batch_size, in_features)
    
    # Your implementation
    custom_layer = LinearLayer(in_features, out_features)
    
    # PyTorch's implementation
    torch_layer = nn.Linear(in_features, out_features)
    
    # Copy weights to make fair comparison
    with torch.no_grad():
        torch_layer.weight.copy_(custom_layer.weight)
        torch_layer.bias.copy_(custom_layer.bias)
    
    # Compare outputs
    custom_output = custom_layer.forward(x)
    torch_output = torch_layer(x)
    
    print("Custom output shape:", custom_output.shape)
    print("Torch output shape:", torch_output.shape)
    print("Outputs are close:", torch.allclose(custom_output, torch_output, atol=1e-6))

if __name__ == "__main__":
    solve()
