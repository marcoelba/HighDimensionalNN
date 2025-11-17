# Transformer module
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_head_attention_layer import MultiHeadCrossAttentionWithWeights


# Standard TransformerEncoderLayer
class TransformerEncoderLayerWithWeights(nn.Module):
    """
    Transformer module that returns attention weights.
    
    Args:
        input_dim (int): Total dimension of the model.
        nheads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Default: 0.0
        dropout_attention (float, optional): Dropout probability. Default: 0.0
        activation: Add activation to output FFN. Default: gelu
    """

    def __init__(self, input_dim, nheads, dim_feedforward, dropout_attention=0.1, dropout=0.1):
        super().__init__()
        # custom attention class, returns the weights in position 1
        self.cross_attn = MultiHeadCrossAttentionWithWeights(
            input_dim,
            nheads,
            dropout_attention,
            bias=True
        )
        # Feed-Forward layers
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layer-Norm
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # dropout layers
        self.dropout1 = nn.Dropout(dropout) # First residual layer
        self.dropout2 = nn.Dropout(dropout) # Dropout for FFN output
        
    def forward(self, src, attn_mask=None):
        # 1. Attention Block (includes output_proj without activation)
        src2 = self.cross_attn(src, src, src, attn_mask=attn_mask)
        src = src + self.dropout1(src2) # Residual connection
        src = self.norm1(src)           # LayerNorm
        
        # 2. FFN Block (This is where the main activation happens!)
        src2 = self.ffn(src) # Activation here
        src = src + self.dropout2(src2) # Residual connection
        src = self.norm2(src)           # LayerNorm
        
        return src
