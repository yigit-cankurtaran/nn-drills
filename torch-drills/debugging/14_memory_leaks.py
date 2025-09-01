"""
DRILL 14: Fix Memory Leaks and Inefficiencies

TASK: This code has memory management issues. Find and fix:

1. Memory leaks from keeping unnecessary gradients
2. Inefficient tensor operations
3. Not using torch.no_grad() when appropriate
4. Creating unnecessary intermediate tensors
5. Not moving tensors to appropriate device

EXPECTED OUTPUT:
- Fixed memory usage
- Faster execution
- Proper device management
- No memory leaks during inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

class MemoryHungryNet(nn.Module):
    def __init__(self):
        super(MemoryHungryNet, self).__init__()
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 1)
    
    def forward(self, x):
        # BUG: Unnecessary intermediate tensor storage
        # no idea what is going on here lol
        intermediate_results = []
        
        x = torch.relu(self.fc1(x))
        intermediate_results.append(x.clone())  # Unnecessary clone
        
        x = torch.relu(self.fc2(x))
        intermediate_results.append(x.clone())  # Unnecessary clone
        
        x = self.fc3(x)
        
        # BUG: Returning unused tensors that keep gradients
        return x, intermediate_results

def problematic_inference(model, data):
    """Inference with memory issues"""
    model.eval()  # BUG: Not using torch.no_grad()
    
    results = []
    for i, batch in enumerate(data):
        # BUG: Creating tensors on wrong device
        batch_gpu = batch.cuda() if torch.cuda.is_available() else batch
        
        # BUG: Not detaching from computation graph
        output, intermediates = model(batch_gpu)
        
        # BUG: Storing tensors with gradients
        results.append(output)
        
        # BUG: Unnecessary tensor operations in loop
        for j in range(len(intermediates)):
            temp = intermediates[j] * 2.0 + 1.0  # Creates new tensors
            temp = temp.cpu()  # Unnecessary device transfers
    
    return results

def problematic_training_loop():
    """Training loop with memory issues"""
    model = MemoryHungryNet()
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Create large dataset
    batch_size = 64
    n_batches = 20
    
    print("Starting problematic training...")
    start_time = time.time()
    
    for epoch in range(5):
        model.train()
        epoch_losses = []
        
        for batch in range(n_batches):
            # BUG: Creating new tensors in every iteration
            X = torch.randn(batch_size, 1000)
            y = torch.randn(batch_size, 1)
            
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()
            
            optimizer.zero_grad()
            
            # BUG: Not handling unused returns properly
            output, intermediates = model(X)
            loss = criterion(output, y)
            
            # BUG: Accumulating gradients in list
            epoch_losses.append(loss)  # Keeps gradients alive
            
            loss.backward()
            optimizer.step()
        
        # BUG: Computing mean of tensor list (keeps all gradients)
        avg_loss = torch.mean(torch.stack(epoch_losses))
        print(f"Epoch {epoch}, Loss: {avg_loss.item():.4f}")
    
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # BUG: Memory-intensive inference
    print("Running inference...")
    test_data = [torch.randn(32, 1000) for _ in range(10)] # test data is here, regression
    # 32 samples, 1000 features each, batch size 32 input size 1000 output size 1
    results = problematic_inference(model, test_data)
    print(f"Inference completed, {len(results)} results")

def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class NeuralNet(nn.Module):
    def __init__(self,input_size=1000,output_size=1):
        # 1000 features and i want single value regression
        # batch sizes are handled by pytorch
        super(NeuralNet,self).__init__()
        self.device = device()
        self.layers = nn.Sequential( # 5 layers, learns patterns but not too deep
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,output_size) # linear on the output bc regression
        )
        self.to(self.device)

    def forward(self,x):
        x = x.to(device())
        return self.layers(x)

# using type hints here
def inference(model, data):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in data: # iterating through every batch
            results.append(model(batch))

    return results

def training_loop():
    model = NeuralNet() # keeping defaults
    model = model.to(device())
        
    

def solve():
    """
    YOUR TASK:
    1. Run the problematic code and observe memory usage
    2. Identify all memory-related issues
    3. Fix them to create an efficient version
    4. Compare memory usage and execution time
    """
    
    print("Running problematic code:")
    problematic_training_loop()
    
    print("\n" + "="*50)
    print("Now implement your memory-efficient fixes:")
    
    # YOUR FIXED CODE HERE
    pass

if __name__ == "__main__":
    solve()
