"""
DRILL 1: Tensor Creation and Basic Operations

TASK: Create the following tensors and perform basic operations:

1. Create a 3x3 tensor filled with zeros
2. Create a 2x4 tensor filled with ones
3. Create a 1D tensor with values [1, 2, 3, 4, 5]
4. Create a 2x3 tensor with random values between 0 and 1
5. Create a 3x3 identity matrix
6. Add the zeros tensor and identity matrix together
7. Multiply the ones tensor by 5

EXPECTED OUTPUT:
- zeros_tensor: tensor of shape (3, 3) filled with 0.0
- ones_tensor: tensor of shape (2, 4) filled with 1.0
- sequence_tensor: tensor([1., 2., 3., 4., 5.])
- random_tensor: tensor of shape (2, 3) with random values
- identity_tensor: 3x3 identity matrix
- added_tensor: sum of zeros_tensor and identity_tensor
- multiplied_tensor: ones_tensor * 5
"""

import torch

def solve():
    # YOUR CODE HERE
    zeros_tensor = torch.zeros(3, 3) # 1 done
    ones_tensor = torch.ones(2, 4) # 2 done
    print(zeros_tensor)
    print(ones_tensor)
    sequence_tensor = torch.tensor([1.,2.,3.,4.,5.]) # 3 done
    print(sequence_tensor)
    print(sequence_tensor.shape)
    random_tensor = torch.rand(size=[2,3]) # 4 done
    print(random_tensor)
    print(random_tensor.shape)
    identity_tensor = torch.eye(3, 3) # 5 done
    print(identity_tensor)
    print(identity_tensor.shape)
    added_tensor = torch.add(zeros_tensor, identity_tensor) # 6 done
    print(added_tensor)
    print(added_tensor.shape)
    multiplied_tensor = torch.mul(ones_tensor, 5) # 7 done
    print(multiplied_tensor)
    print(multiplied_tensor.shape)

    pass

if __name__ == "__main__":
    solve()
