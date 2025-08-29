"""
DRILL 11: Complete Training Loop Implementation

TASK: Implement a complete training loop for a neural network:

1. Create a simple dataset (e.g., linear regression)
2. Define a simple neural network
3. Implement training loop with forward pass, loss calculation, backpropagation
4. Track training metrics (loss, accuracy if applicable)
5. Implement validation loop
6. Add early stopping mechanism

EXPECTED OUTPUT:
- Training loop that decreases loss over epochs
- Proper gradient updates and parameter changes
- Validation metrics tracking
- Early stopping when validation loss stops improving
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        # YOUR CODE HERE
        layers = []
        prev_size = input_size
        hidden_count = 5

        for i in range(hidden_count): # wanna add 5 hidden layers
            layers.append(nn.Linear(prev_size,hidden_size))
            layers.append(nn.LeakyReLU()) # better for regression
            prev_size = hidden_size

        #output, we want regression to output raw vals
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)
        
    
    def forward(self, x):
        # YOUR CODE HERE
        return self.model(x)

def create_linear_dataset(n_samples=1000):
    """Create a simple linear regression dataset"""
    # YOUR CODE HERE
    # Generate data following y = 2x + 1 + noise

    xs=[]
    ys=[]
    
    for x in range(n_samples):
        xs.append(float(x)) # samples should be floats
        noise = random.gauss(0, 0.5)
        y = 2*x + 1 + noise
        ys.append(y)

    # adding extra dimension to make them 2d
    xs = torch.tensor(xs, dtype=torch.float32).unsqueeze(1)
    ys = torch.tensor(ys, dtype=torch.float32).unsqueeze(1) # labels are floats on regression

    return xs, ys

def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        # YOUR CODE HERE
        output = model.forward(batch_x)
        optimizer.zero_grad() # need backward pass even for one epoch
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            # YOUR CODE HERE
            # for validation we only need a forward pass
            output = model.forward(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def solve():
    # Create dataset
    X, y = create_linear_dataset()
    
    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model, loss, optimizer
    model = SimpleNet(input_size=1, hidden_size=10, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    epochs = []
    
    for epoch in range(num_epochs):
        # YOUR CODE HERE
        # Implement the training loop with early stopping
        tl = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(tl)
        vl = validate_epoch(model,val_loader,criterion)
        val_losses.append(vl)
        epochs.append(epoch)

        if epoch % 10 == 0:
            print(f"train loss: {tl} val loss: {vl}")

        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("no improvement in {patience} runs, breaking")
            break
    
    print(f"Training completed at epoch {epoch}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")

    # visualization
    plt.figure(figsize=(10,6))
    plt.plot(epochs, train_losses, "b-", label="Training Loss per Epoch", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Validation Loss per Epoch", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.show()

if __name__ == "__main__":
    solve()
