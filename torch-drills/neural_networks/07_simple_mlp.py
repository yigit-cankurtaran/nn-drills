"""
DRILL 7: Build a Simple Multi-Layer Perceptron

TASK: Create a simple MLP for binary classification:

1. Create an MLP class with configurable hidden layers
2. Use ReLU activation for hidden layers, Sigmoid for output
3. Implement forward pass
4. Create a simple dataset (XOR problem or similar)
5. Test the forward pass with random weights

EXPECTED OUTPUT:
- MLP class that can handle variable number of hidden layers
- Proper forward pass implementation
- Output should be between 0 and 1 (due to sigmoid)
- Network should handle batch processing
"""

import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes (e.g., [64, 32])
            output_size: Size of output layer
        """
        super(SimpleMLP, self).__init__()
        # YOUR CODE HERE
        pass
    
    def forward(self, x):
        # YOUR CODE HERE
        pass

def create_xor_dataset():
    """Create XOR dataset for testing"""
    # YOUR CODE HERE
    pass

def solve():
    # Create XOR dataset
    X, y = create_xor_dataset()
    
    # Create MLP
    mlp = SimpleMLP(input_size=2, hidden_sizes=[4, 4], output_size=1)
    
    # Forward pass
    with torch.no_grad():
        output = mlp(X)
    
    print("Input:")
    print(X)
    print("Target:")
    print(y)
    print("Output:")
    print(output)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    solve()