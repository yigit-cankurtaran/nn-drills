"""
DRILL 8: Understanding Convolutional Layers

TASK: Work with convolutional layers and understand their properties:

1. Create a 2D convolutional layer
2. Apply it to a simple 2D input (like a small image)
3. Experiment with different kernel sizes, padding, and stride
4. Calculate output dimensions manually and verify with actual output
5. Visualize the effect of convolution on a simple pattern

EXPECTED OUTPUT:
- Conv2d layer that processes 2D inputs correctly
- Understanding of how padding and stride affect output size
- Correct output dimension calculations
"""

import torch
import torch.nn as nn

def calculate_conv_output_size(input_size, kernel_size, padding=0, stride=1):
    """
    Calculate the output size of a convolution operation
    Formula: (input_size + 2*padding - kernel_size) / stride + 1
    """
    # YOUR CODE HERE
    return (input_size + 2*padding - kernel_size) / stride + 1
    
def solve():
    # Create a simple 5x5 input (like a tiny image)
    # Add batch and channel dimensions: (batch_size=1, channels=1, height=5, width=5)
    input_tensor = torch.tensor([
        [[[1, 1, 0, 0, 0],
          [1, 1, 0, 0, 0],
          [0, 0, 1, 1, 1],
          [0, 0, 1, 1, 1],
          [0, 0, 1, 1, 1]]]
    ], dtype=torch.float32)
    
    print("Input tensor shape:", input_tensor.shape)
    print("Input tensor:")
    print(input_tensor.squeeze())
    
    # YOUR CODE HERE:
    # 1. Create different conv layers with various parameters
    # 2. Apply them to the input
    # 3. Calculate expected output sizes
    # 4. Compare with actual output sizes
    
    pass

if __name__ == "__main__":
    solve()
