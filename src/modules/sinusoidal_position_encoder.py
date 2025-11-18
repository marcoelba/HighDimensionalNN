# Sinusoidal position encoder
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Add time encoding to array through the Sinusoidal Positional Encoding

    Args:
        d_model (int): Model dimension.
        max_len (int): Maximum length that can be encoded
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Create a vector of positions [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape: [max_len, 1]
        
        # Compute the divisor term for the exponential: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer (a persistent non-trainable parameter)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: [1, max_len, d_model]

    def forward(self, x):
        # x shape is [batch_size, ..., seq_len, d_model]
        # We add a positional encoding for each position in the sequence
        x = x + self.pe[:, :x.size(-2), :]
        return x


if __name__ == "main":
    # Example usage
    d_model = 512
    seq_len = 10
    batch_size = 2

    input_tensor = torch.randn(batch_size, seq_len, d_model)
    pos_encoder = SinusoidalPositionalEncoding(d_model)
    output_with_pos = pos_encoder(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Positional encoding shape:", pos_encoder.pe[:, :seq_len, :].shape)
    print("Output shape:", output_with_pos.shape) # Same as input
