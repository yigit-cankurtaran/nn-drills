"""
DRILL 3: Tensor Reshaping and Dimension Manipulation

TASK: Given a 1D tensor with 12 elements, perform the following operations:

1. Reshape it to a 3x4 tensor
2. Reshape it to a 2x6 tensor
3. Reshape it to a 4x3 tensor
4. Add a new dimension at position 0 (unsqueeze)
5. Remove dimension of size 1 (squeeze)
6. Transpose the 3x4 tensor
7. Flatten any tensor back to 1D

EXPECTED OUTPUT:
- tensor_3x4: tensor of shape (3, 4)
- tensor_2x6: tensor of shape (2, 6)
- tensor_4x3: tensor of shape (4, 3)
- unsqueezed: tensor with added dimension
- squeezed: tensor with removed singleton dimensions
- transposed: transposed version of tensor_3x4
- flattened: 1D version of any tensor
"""

import torch

def solve():
    # Given 1D tensor
    tensor_1d = torch.arange(1, 13, dtype=torch.float32)
    
    print("Original 1D tensor:")
    print(tensor_1d)
    print("Shape:", tensor_1d.shape)
    
    # YOUR CODE HERE
    tensor_copy = torch.clone(tensor_1d) # ensuring og tensor stays the same
    tensor_3x4 = torch.reshape(tensor_copy, [3,4])
    tensor_2x6 = torch.reshape(tensor_copy, [2,6])
    tensor_4x3 = torch.reshape(tensor_copy, [4,3])
    unsqueezed = torch.unsqueeze(tensor_copy, 0)
    squeezed = torch.squeeze(unsqueezed)
    transposed = tensor_3x4.T
    flattened = torch.flatten(tensor_2x6) # tensor_copy anyway

if __name__ == "__main__":
    solve()
