"""
DRILL 6: Implementing Activation Functions

TASK: Implement common activation functions from scratch:

1. ReLU: max(0, x)
2. Sigmoid: 1 / (1 + exp(-x))
3. Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
4. Leaky ReLU: max(0.01*x, x)
5. Swish/SiLU: x * sigmoid(x)
6. Compare with PyTorch's built-in implementations

EXPECTED OUTPUT:
- All activation functions implemented correctly
- Outputs should match PyTorch's implementations within tolerance
- Functions should handle negative and positive inputs properly
"""

import torch
import torch.nn.functional as F

def custom_relu(x):
    # YOUR CODE HERE
    pass

def custom_sigmoid(x):
    # YOUR CODE HERE
    pass

def custom_tanh(x):
    # YOUR CODE HERE
    pass

def custom_leaky_relu(x, negative_slope=0.01):
    # YOUR CODE HERE
    pass

def custom_swish(x):
    # YOUR CODE HERE
    pass

def solve():
    # Test input with various values
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    
    print("Input tensor:", x)
    print()
    
    # Test ReLU
    custom_relu_out = custom_relu(x)
    torch_relu_out = F.relu(x)
    print("ReLU - Custom:", custom_relu_out)
    print("ReLU - Torch: ", torch_relu_out)
    print("ReLU - Close: ", torch.allclose(custom_relu_out, torch_relu_out))
    print()
    
    # YOUR CODE HERE: Test other activation functions
    pass

if __name__ == "__main__":
    solve()