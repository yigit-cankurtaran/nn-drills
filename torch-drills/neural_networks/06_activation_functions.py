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

# using torch everywhere bc x can be multidimensional tensors
# checking simply for 1 variable brings runtime bool errors
def custom_relu(x):
    # YOUR CODE HERE
    return torch.clamp(x, min=0)

def custom_sigmoid(x):
    # YOUR CODE HERE
    return 1 / (1 + torch.exp(-x))

def custom_tanh(x):
    # YOUR CODE HERE
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    #exponentials should work okay with negative vals

def custom_leaky_relu(x, negative_slope=0.01):
    # YOUR CODE HERE
    return torch.maximum(x, negative_slope*x)

def custom_swish(x):
    # YOUR CODE HERE
    return torch.mul(x, custom_sigmoid(x))
    

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
    custom_sigmoid_out =custom_sigmoid(x)
    torch_sigmoid_out = F.sigmoid(x)
    print("sigmoid - Custom:", custom_sigmoid_out)
    print("sigmoid - Torch: ", torch_sigmoid_out)
    print("sigmoid - Close: ", torch.allclose(custom_sigmoid_out, torch_sigmoid_out))
    print()

    custom_tanh_out =custom_tanh(x)
    torch_tanh_out = F.tanh(x)
    print("tanh - Custom:", custom_tanh_out)
    print("tanh - Torch: ", torch_tanh_out)
    print("tanh - Close: ", torch.allclose(custom_tanh_out, torch_tanh_out))
    print()
    
    custom_leaky_relu_out =custom_leaky_relu(x)
    torch_leaky_relu_out = F.leaky_relu(x)
    print("leaky_relu - Custom:", custom_leaky_relu_out)
    print("leaky_relu - Torch: ", torch_leaky_relu_out)
    print("leaky_relu - Close: ", torch.allclose(custom_leaky_relu_out, torch_leaky_relu_out))
    print()

    custom_swish_out =custom_swish(x)
    torch_swish_out = F.silu(x)
    print("swish - Custom:", custom_swish_out)
    print("swish - Torch: ", torch_swish_out)
    print("swish - Close: ", torch.allclose(custom_swish_out, torch_swish_out))
    print()


if __name__ == "__main__":
    solve()
