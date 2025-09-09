"""
DRILL 18: Fill in the Blanks - Custom Autograd Function

TASK: Complete the implementation of a custom autograd function for element-wise square operation.
Fill in the blanks marked with # FILL IN: comments.

EXPECTED OUTPUT:
- Working custom autograd function
- Correct forward and backward passes
- Gradients computed correctly
"""

import torch
from torch.autograd import Function

class SquareFunction(Function):
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: compute input^2
        Args:
            ctx: context object for storing information for backward pass
            input: input tensor
        Returns:
            output: input^2
        """
        # FILL IN: Store input for backward pass
        ctx.save_for_backward(input)
        
        # FILL IN: Compute and return input squared
        return input**2
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient
        Args:
            ctx: context object containing saved tensors
            grad_output: gradient flowing back from next layer
        Returns:
            grad_input: gradient with respect to input
        """
        # FILL IN: Retrieve saved input
        input, = ctx.saved_tensors
        
        # FILL IN: Compute gradient: d/dx(x^2) = 2x
        grad_input = grad_output * 2 * input # 2x = 2*input
        
        return grad_input

# Create convenient function wrapper
def square(input):
    return SquareFunction.apply(input)

def solve():
    # Test the custom square function
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    
    print("Input:", x)
    
    # FILL IN: Apply custom square function
    y = square(x)
    print("Output (y = x^2):", y)
    
    # FILL IN: Compute loss (sum of squares for simplicity)
    loss = torch.sum(y)
    print("Loss (sum of y):", loss)
    
    # Backward pass
    loss.backward()
    
    print("Gradients (dy/dx):", x.grad)
    
    # Verify correctness
    # For y = x^2 and loss = sum(y), gradient should be 2x
    expected_grad = 2 * x.detach()
    print("Expected gradients:", expected_grad)
    
    # FILL IN: Check if gradients are correct
    print("Gradients correct:", torch.allclose(x.grad, expected_grad))
    
    # Test with PyTorch's built-in square for comparison
    x2 = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y2 = torch.square(x2)
    loss2 = torch.sum(y2)
    loss2.backward()
    
    print("PyTorch gradients:", x2.grad)
    print("Custom vs PyTorch gradients match:", torch.allclose(x.grad, x2.grad))
    
    # Test with matrix input
    print("\n=== Testing with matrix input ===")
    X = torch.randn(3, 4, requires_grad=True)
    
    # FILL IN: Apply custom square to matrix
    Y = square(X)
    matrix_loss = torch.sum(Y)
    matrix_loss.backward()
    
    print("Matrix input shape:", X.shape)
    print("Matrix output shape:", Y.shape)
    print("Matrix gradients shape:", X.grad.shape)
    
    # Verify matrix gradients
    expected_matrix_grad = 2 * X.detach()
    print("Matrix gradients correct:", torch.allclose(X.grad, expected_matrix_grad))

if __name__ == "__main__":
    solve()
