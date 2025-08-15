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
    element = tensor[2,1] #confused, do i use 1,0 or 2,1
    print(element)
    print(element.shape)
    second_row = tensor[1]
    print(second_row)
    print(second_row.shape)
    last_column = tensor[:, -1] # all rows, last column only
    print(last_column)
    print(last_column.shape)
    submatrix = tensor[:2, :2]
    print(submatrix)
    print(submatrix.shape)
    every_other = tensor[0, ::2]
    print(every_other)
    print(every_other.shape)
    modified_tensor = tensor.clone()
    modified_tensor[:, 2] = 99
    print(modified_tensor)
    print(modified_tensor.shape)
    
    pass

if __name__ == "__main__":
    solve()
