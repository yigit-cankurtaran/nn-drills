"""
DRILL 2: Tensor Indexing and Slicing

TASK: Given a 4x4 tensor, perform the following indexing operations:

1. Extract the element at position (2, 1)
2. Extract the second row
3. Extract the last column
4. Extract a 2x2 submatrix from the top-left corner
5. Extract every other element from the first row
6. Set all elements in the third column to 99

EXPECTED OUTPUT:
- element: single tensor value at (2, 1)
- second_row: tensor of shape (4,)
- last_column: tensor of shape (4,)
- submatrix: tensor of shape (2, 2)
- every_other: tensor with every other element from first row
- modified_tensor: original tensor with third column set to 99
"""

import torch

def solve():
    # Given tensor
    tensor = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=torch.float32)
    
    print("Original tensor:")
    print(tensor)
    
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    solve()