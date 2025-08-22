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
    output =  (input_size + 2*padding - kernel_size) // stride + 1
    return output    

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
    
    print("Input tensor shape:", input_tensor.shape) #[1,1,5,5]
    print("Input tensor:")
    print(input_tensor.squeeze()) #prints with shape [1,5,5]
    
    # YOUR CODE HERE:
    # 1. Create different conv layers with various parameters
    # 2. Apply them to the input
    # 3. Calculate expected output sizes
    # 4. Compare with actual output sizes

    clone_tensor = torch.clone(input_tensor) #4d tensor
    # the shape is [1,1,5,5] bc those are batch and channel dimensions, we need it

    kernel_size = 3
    padding = 2
    stride = 2

    conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=kernel_size)
    conv_tensor = conv(clone_tensor)
    conv_output = calculate_conv_output_size(input_size=clone_tensor.size(2),kernel_size=kernel_size) # passing in height and kernel
    print(f"conv tensor is {conv_tensor}")
    if (conv_output == conv_tensor.shape[2]):
        print("convolution results match")

    conv_with_stride = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=kernel_size,stride=stride)
    stride_tensor = conv_with_stride(clone_tensor)
    stride_output = calculate_conv_output_size(clone_tensor.size(2),kernel_size=kernel_size,stride=stride)
    print(f"clone_tensor after stride is {stride_tensor}")
    if (stride_output == stride_tensor.shape[2]):
        print("stride results match")

    conv_with_padding = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding)
    padding_tensor = conv_with_padding(clone_tensor)
    padding_output = calculate_conv_output_size(clone_tensor.size(2),kernel_size=kernel_size,padding=padding)
    print(f"clone_tensor after padding is {padding_tensor}")
    if (padding_output == padding_tensor.shape[2]):
        print("padding results match")

    conv_with_padding_and_stride = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding,stride=stride)
    padding_and_stride_tensor = conv_with_padding_and_stride(clone_tensor)
    padding_and_stride_output = calculate_conv_output_size(clone_tensor.size(2),kernel_size=kernel_size,stride=stride,padding=padding)    
    print(f"clone_tensor after padding_and_stride is {padding_and_stride_tensor}")
    if (padding_and_stride_output == padding_and_stride_tensor.shape[2]):
        print("padding_and_stride results match")

if __name__ == "__main__":
    solve()
