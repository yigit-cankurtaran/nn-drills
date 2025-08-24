"""
DRILL 9: Implementing Loss Functions

TASK: Implement common loss functions from scratch:

1. Mean Squared Error (MSE) for regression
2. Binary Cross Entropy for binary classification
3. Cross Entropy for multi-class classification
4. Compare with PyTorch's built-in implementations
5. Test with appropriate sample data

EXPECTED OUTPUT:
- All loss functions implemented correctly
- Outputs match PyTorch's implementations within tolerance
- Proper handling of edge cases (like log(0))
"""

import torch
import torch.nn.functional as F

def custom_mse_loss(predictions, targets):
    """Mean Squared Error Loss"""
    # YOUR CODE HERE
    # square of the differences
    sum = 0
    n = 0
    for p,t in zip(predictions, targets):
        sum += (t - p) ** 2
        n += 1

    return sum / n
        

def custom_binary_cross_entropy(predictions, targets):
    """Binary Cross Entropy Loss (with logits)"""
    # YOUR CODE HERE
    predictions = torch.clamp(predictions, 1e-7, 1-1e-7)
    n = len(predictions)
    sum = 0
    for p,t in zip(predictions,targets):
        intermediate1 = t * torch.log(p)
        intermediate2 = (1-t) * torch.log(1-p)
        sum += intermediate1 + intermediate2

    return -sum/n

def custom_cross_entropy(logits, targets):
    """Cross Entropy Loss"""
    # YOUR CODE HERE
    # Hint: Use log_softmax and gather operations
    probabilities = torch.log_softmax(logits, dim=-1)

    batch_size = logits.shape[0] # returns 3, gets number of samples in batch
    targets_reshaped = targets.view(batch_size,1) # reshape to 3,1 for gather

    # select log prob at the index specified by target
    selected_probs = probabilities.gather(1, targets_reshaped).squeeze(1)

    return -selected_probs.mean()

def solve():
    # Test MSE Loss
    print("=== Testing MSE Loss ===")
    pred_regression = torch.tensor([1.0, 2.0, 3.0])
    target_regression = torch.tensor([1.1, 1.9, 3.2])
    
    custom_mse = custom_mse_loss(pred_regression, target_regression)
    torch_mse = F.mse_loss(pred_regression, target_regression)
    
    print(f"Custom MSE: {custom_mse:.6f}")
    print(f"Torch MSE:  {torch_mse:.6f}")
    print(f"Close: {torch.allclose(custom_mse, torch_mse)}")
    print()
    
    # Test Binary Cross Entropy
    print("=== Testing Binary Cross Entropy ===")
    # YOUR CODE HERE: Create test data and compare implementations
    pred_binary = torch.tensor([0.8, 0.3, 0.9])
    target_binary = torch.tensor([1.0, 0.0, 1.0])

    custom_bce = custom_binary_cross_entropy(pred_binary, target_binary)
    torch_bce = F.binary_cross_entropy(pred_binary, target_binary)
    
    print(f"Custom BCE: {custom_bce:.6f}")
    print(f"Torch BCE:  {torch_bce:.6f}")
    print(f"Close: {torch.allclose(custom_bce, torch_bce)}")
    print()
    
    # Test Cross Entropy
    print("=== Testing Cross Entropy ===")
    # YOUR CODE HERE: Create test data and compare implementations
    matrix_bin = torch.tensor([[0.1, 0.8, 0.1],
                           [0.7, 0.2, 0.1],
                           [0.1, 0.1, 0.8]])
    class_indices = torch.tensor([1,0,2], dtype=torch.long) # long for class types
    custom_ce = custom_cross_entropy(matrix_bin, class_indices)
    torch_ce = F.cross_entropy(matrix_bin, class_indices)

    print(f"Custom CE: {custom_ce.item():.6f}")
    print(f"Torch CE:  {torch_ce:.6f}")
    print(f"Close: {torch.allclose(custom_ce, torch_ce)}")
    print()


if __name__ == "__main__":
    solve()
