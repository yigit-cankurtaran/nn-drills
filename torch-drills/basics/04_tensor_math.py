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

    # element-wise 4 operations
    addeda = torch.add(a, 5)
    subtracteda = torch.sub(addeda, 5)
    multiplieda = torch.mul(subtracteda, 3)
    divideda = torch.div(multiplieda, 3)

    # question 2
    matmulbc = torch.mm(b, c) # picked b@c b is 2x3 and c is 3x2

    # question 3
    meanb = torch.mean(b)
    sumb = torch.sum(b)
    maxb = torch.max(b)
    minb = torch.min(b)

    # question 4
    sinc = torch.sin(c)
    cosc = torch.cos(c)
    expc = torch.exp(c)
    logc = torch.log(c) #natural logarithm

    
    # question 5
    to_add = torch.asarray([1.,2.,3.]) # needs to be length 3 bc a has 3 cols
    tensoradd = torch.add(a, to_add)
    
    pass

if __name__ == "__main__":
    solve()
