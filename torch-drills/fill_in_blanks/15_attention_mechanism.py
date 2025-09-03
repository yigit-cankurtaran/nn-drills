"""
DRILL 15: Fill in the Blanks - Attention Mechanism

TASK: Complete the missing parts of a simple attention mechanism implementation.
Fill in the blanks marked with # FILL IN: comments.

EXPECTED OUTPUT:
- Working attention mechanism
- Proper attention weights (sum to 1)
- Context vector computed correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.hidden_size = hidden_size
        # FILL IN: Create linear layer for computing attention scores
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys, values):
        """
        Args:
            query: (batch_size, hidden_size)
            keys: (batch_size, seq_len, hidden_size)  
            values: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size)
            attention_weights: (batch_size, seq_len)
        """
        batch_size, seq_len, hidden_size = keys.size()
        
        # FILL IN: Expand query to match keys dimensions
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        # -1 = don't change this dimension's size
        # we only need to add a new dimension(unsqueeze) and then expand it to seq_len
        
        # FILL IN: Compute attention scores using dot product
        scores = torch.sum(query_expanded * keys, dim=2) # sum along the hidden_size dimension
        
        # FILL IN: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1) # normalize across seq_len dimension
        
        # FILL IN: Compute context vector as weighted sum of values
        context = torch.sum(attention_weights.unsqueeze(2) * values, dim=1) # sum along seq_len        
        return context, attention_weights

def solve():
    # Test the attention mechanism
    batch_size, seq_len, hidden_size = 2, 4, 8
    
    # Create test data
    query = torch.randn(batch_size, hidden_size)
    keys = torch.randn(batch_size, seq_len, hidden_size)
    values = torch.randn(batch_size, seq_len, hidden_size)
    
    # FILL IN: Create attention module
    attention = SimpleAttention(hidden_size) # the init method takes only hidden_size
    
    # Forward pass
    context, weights = attention(query, keys, values)
    
    print("Query shape:", query.shape)
    print("Keys shape:", keys.shape)
    print("Values shape:", values.shape)
    print("Context shape:", context.shape)
    print("Attention weights shape:", weights.shape)
    
    # FILL IN: Check that attention weights sum to 1
    weights_sum = torch.sum(weights, dim=1)
    print("Attention weights sum:", weights_sum)
    print("Weights sum to 1:", torch.allclose(weights_sum, torch.ones_like(weights_sum)))

if __name__ == "__main__":
    solve()
