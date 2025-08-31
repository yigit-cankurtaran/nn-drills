"""
DRILL 13: Debug Gradient-Related Issues

TASK: This code has gradient-related problems. Identify and fix:

1. Vanishing/exploding gradients
2. Incorrect gradient accumulation
3. Missing requires_grad
4. In-place operations breaking gradients
5. Gradient clipping issues

EXPECTED OUTPUT:
- Fixed gradient flow
- Proper parameter updates
- Stable training without gradient explosions
"""

import torch
import torch.nn as nn
import torch.optim as optim

class ProblematicNet(nn.Module):
    def __init__(self):
        super(ProblematicNet, self).__init__()
        # BUG: Poor weight initialization leading to vanishing gradients
        self.layers = nn.ModuleList([ # shapes look reasonable
            nn.Linear(10, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 1)
        ])
        
        # BUG: Initialize weights poorly
        for layer in self.layers:
            layer.weight.data.fill_(0.01)  # Very small weights
            # we can fix this through xavier initialization
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            # BUG: Using sigmoid everywhere causes vanishing gradients
            x = torch.sigmoid(x)
            # scaling down activation too much, creates vanishing activations and gradients
            x *= 0.1 # completely breaks training loop
        
        x = self.layers[-1](x)
        return x

def problematic_training():
    # Create some dummy data
    X = torch.randn(100, 10)  # BUG: Missing requires_grad for input
    y = torch.randn(100, 1)
    
    model = ProblematicNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # BUG: Not clearing gradients properly
        if epoch % 2 == 0:  # Only clear gradients every other epoch
            optimizer.zero_grad()
        # gonna need to do this before every backward pass
        
        loss.backward()
        
        # BUG: Gradient clipping applied incorrectly
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
        # needs larger max_norm, might make it 0.5 myself
        
        optimizer.step()
        
        if epoch % 20 == 0:
            # Check gradients
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Grad Norm: {total_grad_norm:.6f}')

class GoodNet(nn.Module):
    def __init__(self):
        super(GoodNet,self).__init__()
        self.model= nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(), # ReLU activation in between layers
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
            # deleted sigmoid because it's a regression problem
            # regression problems need linear output
            )

        print(f"model:\n{self.model}\n")

        # xavier uniform
        for module in self.model:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X):
        return self.model(X)

def good_training():
    X = torch.randn(100, 10, requires_grad=True) # torch will track gradients
    y = torch.randn(100, 1) # labels

    model = GoodNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        if epoch % 10 == 0:
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Grad Norm: {total_grad_norm:.6f}')
        
def solve():
    """
    YOUR TASK: 
    1. Run the problematic code and observe the issues
    2. Identify all gradient-related problems
    3. Fix them and create a working version
    """
    
    print("Running problematic network:")
    # problematic_training()
    
    print("\n" + "="*50)
    print("Now implement your fixes:")
    
    # YOUR FIXED CODE HERE
    print("good training:")
    good_training()

if __name__ == "__main__":
    solve()
