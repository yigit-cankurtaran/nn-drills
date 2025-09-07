"""
DRILL 16: Fill in the Blanks - Batch Normalization

TASK: Complete the implementation of batch normalization from scratch.
Fill in the blanks marked with # FILL IN: comments.

EXPECTED OUTPUT:
- Working batch normalization layer
- Proper normalization (mean≈0, std≈1)
- Learnable parameters working correctly
"""

import torch
import torch.nn as nn

class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(CustomBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # FILL IN: Create learnable parameters gamma and beta
        # look up how this all works
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # FILL IN: Create running statistics (not trainable)
        self.register_buffer('running_mean', torch.zeros(num_features))
        # register_buffer is used for stuff that is not a parameter but part of state
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_features)
        Returns:
            normalized x: (batch_size, num_features)
        """
        if self.training:
            # FILL IN: Compute batch statistics
            batch_mean = torch.mean(x, dim=0) # for batch statistics, we compute across batch
            batch_var_biased = torch.var(x, dim=0, unbiased=False)  # for normalization
            batch_var_unbiased = torch.var(x, dim=0, unbiased=True)  # for running stats
            
            # FILL IN: Update running statistics
            # exponential moving average update formula
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var_unbiased
            
            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var_biased
        else:
            # FILL IN: Use running statistics during inference
            mean = self.running_mean
            var = self.running_var
        
        # FILL IN: Normalize the input
        x_normalized = (x - mean) / torch.sqrt(var + self.eps) # epsilon
        
        # FILL IN: Apply learnable transformation
        output = self.gamma * x_normalized + self.beta
        
        return output

def solve():
    # Test the custom batch norm
    batch_size, num_features = 32, 10
    
    # Create test data with non-zero mean and non-unit variance
    x = torch.randn(batch_size, num_features) * 3 + 5
    
    print("Input statistics:")
    print(f"Mean: {x.mean(dim=0)}")
    print(f"Std: {x.std(dim=0)}")
    
    # FILL IN: Create custom batch norm layer
    custom_bn = CustomBatchNorm1d(num_features) #the rest are defaults
    
    # FILL IN: Create PyTorch's batch norm for comparison
    torch_bn = nn.BatchNorm1d(num_features) #the rest are defaults again
    
    # Copy parameters to make fair comparison
    with torch.no_grad():
        torch_bn.weight.copy_(custom_bn.gamma)
        torch_bn.bias.copy_(custom_bn.beta)
    
    # Forward pass in training mode
    custom_bn.train()
    torch_bn.train()
    
    custom_output = custom_bn(x)
    torch_output = torch_bn(x)
    
    print("\nOutput statistics (training mode):")
    print(f"Custom BN mean: {custom_output.mean(dim=0)}")
    print(f"Custom BN std: {custom_output.std(dim=0)}")
    print(f"PyTorch BN mean: {torch_output.mean(dim=0)}")
    print(f"PyTorch BN std: {torch_output.std(dim=0)}")
    
    # FILL IN: Check if outputs are close
    print(f"Outputs are close: {torch.allclose(custom_output, torch_output, atol=1e-5)}")
    
    # Test inference mode
    custom_bn.eval()
    torch_bn.eval()
    
    custom_output_eval = custom_bn(x)
    torch_output_eval = torch_bn(x)
    
    print(f"Inference outputs are close: {torch.allclose(custom_output_eval, torch_output_eval, atol=1e-5)}")

if __name__ == "__main__":
    solve()
