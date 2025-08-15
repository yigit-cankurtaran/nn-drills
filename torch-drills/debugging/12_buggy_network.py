"""
DRILL 12: Debug a Buggy Neural Network

TASK: The following code has several bugs. Find and fix them:

1. Tensor dimension mismatches
2. Incorrect loss function usage
3. Missing gradient computation
4. Wrong activation functions
5. Optimizer not updating parameters

EXPECTED OUTPUT:
- Fixed code that trains without errors
- Network should learn the XOR function
- Loss should decrease over training iterations

BUGS TO FIND:
- At least 5 different types of bugs are hidden in this code
"""

import torch
import torch.nn as nn
import torch.optim as optim

class BuggyXORNet(nn.Module):
    def __init__(self):
        super(BuggyXORNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 5)  # BUG: Wrong input size
        self.fc3 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # BUG: Wrong activation
        x = self.fc3(x)
        return torch.tanh(x)  # BUG: Wrong output activation

def create_xor_data():
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    return X, y

def train_buggy_network():
    # Create data
    X, y = create_xor_data()
    
    # Create network
    model = BuggyXORNet()
    criterion = nn.MSELoss()  # BUG: Wrong loss function for this problem
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Training loop
    for epoch in range(1000):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        # BUG: Missing loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    # Test final predictions
    with torch.no_grad():
        predictions = model(X)
        print("Final predictions:")
        for i, (input_val, target, pred) in enumerate(zip(X, y, predictions)):
            print(f"Input: {input_val.numpy()}, Target: {target.item():.1f}, Prediction: {pred.item():.4f}")

def solve():
    """
    YOUR TASK: Fix all the bugs in the code above and make it work properly.
    The network should learn the XOR function successfully.
    """
    print("Running buggy network (should have errors):")
    try:
        train_buggy_network()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50)
    print("Now fix the bugs and implement the corrected version below:")
    
    # YOUR FIXED CODE HERE
    pass

if __name__ == "__main__":
    solve()