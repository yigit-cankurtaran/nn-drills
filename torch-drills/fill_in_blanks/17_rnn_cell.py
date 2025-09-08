"""
DRILL 17: Fill in the Blanks - RNN Cell Implementation

TASK: Complete the implementation of a basic RNN cell from scratch.
Fill in the blanks marked with # FILL IN: comments.

EXPECTED OUTPUT:
- Working RNN cell
- Proper hidden state propagation
- Correct output dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # FILL IN: Create weight matrices for input-to-hidden and hidden-to-hidden connections
        self.W_ih = nn.Linear(input_size, hidden_size)  # input to hidden
        self.W_hh = nn.Linear(hidden_size, hidden_size)  # hidden to hidden
    
    def forward(self, input, hidden):
        """
        Args:
            input: (batch_size, input_size)
            hidden: (batch_size, hidden_size)
        Returns:
            new_hidden: (batch_size, hidden_size)
        """
        # FILL IN: Compute new hidden state
        # Formula: h_new = tanh(W_ih @ input + W_hh @ hidden)
        new_hidden = torch.tanh(self.W_ih(input) + self.W_hh(hidden))
        
        return new_hidden

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # FILL IN: Create RNN cell
        self.rnn_cell = CustomRNNCell(input_size, hidden_size)
    
    def forward(self, input, hidden=None):
        """
        Args:
            input: (batch_size, seq_len, input_size)
            hidden: (batch_size, hidden_size) or None
        Returns:
            output: (batch_size, seq_len, hidden_size)
            hidden: (batch_size, hidden_size)
        """
        batch_size, seq_len, input_size = input.size()
        
        # FILL IN: Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        
        # FILL IN: Process sequence step by step
        for t in range(seq_len):
            # Get input at time step t
            input_t = input[:, t, :]
            
            # Update hidden state
            hidden = self.rnn_cell(input_t, hidden)
            
            # Store output
            outputs.append(hidden)
        
        # FILL IN: Stack outputs along time dimension
        output = torch.stack(outputs, dim=1) # stack expects list, not unpacked list
        
        return output, hidden

def solve():
    # Test the custom RNN
    batch_size, seq_len, input_size, hidden_size = 3, 5, 4, 8
    
    # Create test data
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Input shape:", x.shape)
    
    # FILL IN: Create custom RNN
    custom_rnn = SimpleRNN(_____, _____)
    
    # FILL IN: Create PyTorch's RNN for comparison
    torch_rnn = nn.RNN(_____, _____, batch_first=True)
    
    # Forward pass
    custom_output, custom_hidden = custom_rnn(x)
    torch_output, torch_hidden = torch_rnn(x)
    
    print("Custom RNN output shape:", custom_output.shape)
    print("Custom RNN hidden shape:", custom_hidden.shape)
    print("PyTorch RNN output shape:", torch_output.shape)
    print("PyTorch RNN hidden shape:", torch_hidden.squeeze(0).shape)
    
    # Test with initial hidden state
    init_hidden = torch.randn(batch_size, hidden_size)
    custom_output2, custom_hidden2 = custom_rnn(x, init_hidden)
    
    print("With initial hidden - output shape:", custom_output2.shape)
    print("With initial hidden - final hidden shape:", custom_hidden2.shape)
    
    # FILL IN: Verify that the last output equals the final hidden state
    last_output = custom_output[:, -1, :]
    print("Last output equals final hidden:", torch.allclose(last_output, _____))

if __name__ == "__main__":
    solve()
