"""
DRILL 4: Tensor Mathematical Operations

TASK: Perform various mathematical operations on tensors:

1. Element-wise addition, subtraction, multiplication, division
2. Matrix multiplication between two 2D tensors
3. Calculate mean, sum, max, min of a tensor
4. Apply mathematical functions: sin, cos, exp, log
5. Broadcasting: add a 1D tensor to each row of a 2D tensor

EXPECTED OUTPUT:
- All operations should work correctly with proper shapes
- Mathematical operations should preserve numerical accuracy
- Broadcasting should work without explicit loops
"""

import torch
import math

def solve():
    # Create test tensors
    a = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    
    b = torch.tensor([[7.0, 8.0, 9.0],
                      [10.0, 11.0, 12.0]])
    
    c = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
    
    bias = torch.tensor([0.1, 0.2, 0.3])
    
    print("Tensor a:")
    print(a)
    print("Tensor b:")
    print(b)
    print("Tensor c:")
    print(c)
    print("Bias tensor:")
    print(bias)
    
    # YOUR CODE HERE
    # Implement all the required operations
    d = torch.add(a, 5)
    print(d)
    pass

if __name__ == "__main__":
    solve()
