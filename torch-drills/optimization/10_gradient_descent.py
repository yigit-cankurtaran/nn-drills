"""
DRILL 10: Manual Gradient Descent Implementation

TASK: Implement gradient descent from scratch:

1. Create a simple quadratic function f(x) = (x - 3)^2 + 5
2. Implement manual gradient descent to find the minimum
3. Compare with automatic differentiation results
4. Implement SGD, Adam optimizer basics
5. Plot the optimization trajectory

EXPECTED OUTPUT:
- Manual gradient descent finds minimum at x=3
- Convergence to the correct minimum value
- Comparison with PyTorch's automatic differentiation
"""

import torch
import matplotlib.pyplot as plt

def quadratic_function(x):
    """f(x) = (x - 3)^2 + 5"""
    return (x - 3)**2 + 5

def quadratic_gradient(x):
    """df/dx = 2(x - 3)"""
    # YOUR CODE HERE
    return 2*(x-3)

def manual_gradient_descent(start_x, learning_rate, num_steps):
    """
    Implement gradient descent manually
    Returns: history of x values and function values
    """
    x_history = []
    f_history = []
    
    x = start_x
    for step in range(num_steps):
        # YOUR CODE HERE
        x_history.append(x)
        y = quadratic_function(x)
        f_history.append(y)
        grad = quadratic_gradient(x) # don't forget to get the actual gradient!
        x = x - (learning_rate * grad)
    
    return x_history, f_history

def torch_gradient_descent(start_x, learning_rate, num_steps):
    """
    Use PyTorch's automatic differentiation
    """
    x = torch.tensor([start_x], requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=learning_rate)
    
    x_history = []
    f_history = []
    
    for step in range(num_steps):
        # YOUR CODE HERE
        x_history.append(x.item())
        y = (x-3)**2 + 5 # keeping y as a tensor
        f_history.append(y.item())
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return x_history, f_history

def adam_gradient_descent(start_x, learning_rate, num_steps):
    """
    gonna use adam here
    """
    # brackets bc we want to create a 1D tensor from a scalar
    x = torch.tensor([start_x], requires_grad=True)
    # momentums of Adam can be changed through betas, not gonna do it here bc basics
    optimizer = torch.optim.Adam(params=[x], lr=learning_rate)

    x_history = []
    f_history = []

    for step in range(num_steps):
        x_history.append(x.item())
        y = (x-3)**2 + 5 # keeping y as a tensor
        f_history.append(y.item())
        optimizer.zero_grad()
        y.backward()
        optimizer.step()

    return x_history, f_history

def solve():
    start_x = 0.0
    learning_rate = 0.1
    num_steps = 500
    
    # Manual gradient descent
    manual_x_hist, manual_f_hist = manual_gradient_descent(start_x, learning_rate, num_steps)
    
    # PyTorch gradient descent
    torch_x_hist, torch_f_hist = torch_gradient_descent(start_x, learning_rate, num_steps)

    # adam_gradient_descent
    adam_x_hist, adam_f_hist = adam_gradient_descent(start_x, learning_rate, num_steps)
    
    print(f"Manual GD final x: {manual_x_hist[-1]:.6f}")
    print(f"Torch GD final x:  {torch_x_hist[-1]:.6f}")
    print("Target x: 3.0")
    print(f"Manual GD final f(x): {manual_f_hist[-1]:.6f}")
    print(f"Torch GD final f(x):  {torch_f_hist[-1]:.6f}")
    print("Target f(x): 5.0")
    print(f"Adam GD final x:  {adam_x_hist[-1]:.6f}")
    print(f"Adam GD final f(x):  {adam_f_hist[-1]:.6f}")

    # TODO plot the optimization graphics

if __name__ == "__main__":
    solve()
