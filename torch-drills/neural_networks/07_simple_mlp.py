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

def device():
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.mps.is_available():
        dev = torch.device("mps")

    print(f"device is {dev}")
    return dev

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
        self.model = nn.Sequential(
            #input
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            # hidden layers
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.Sigmoid()
        ).to(device())
    
    def forward(self, x):
        # YOUR CODE HERE
        self.output = self.model(x)
        return self.output

def create_xor_dataset():
    """Create XOR dataset for testing"""
    xor_vals = [[0.,0.],
                [0.,1.],
                [1.,0.],
                [1.,1.]]
    xor_ans = [[0],
            [1],
            [1],
            [0]]
    xor_vals = torch.tensor(xor_vals).to(device())
    xor_ans = torch.tensor(xor_ans).to(device())
    return xor_vals, xor_ans

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
