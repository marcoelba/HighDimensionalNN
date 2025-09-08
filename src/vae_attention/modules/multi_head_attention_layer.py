# custom scaled dot attention layer with multi-head
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttentionWithWeights(nn.Module):
    """
    Multi-head attention module that returns attention weights for visualization.
    
    Args:
        input_dim (int): Total dimension of the model.
        nheads (int): Number of attention heads.
        dropout_attention (float, optional): Attention Dropout probability. Default: 0.1
        bias (bool, optional): Add bias to linear projections. Default: True
    """
    def __init__(self, input_dim, nheads, dropout_attention=0.1, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.nheads = nheads
        self.dropout_attention = dropout_attention
        
        # Ensure d_model is divisible by nhead
        assert input_dim % nheads == 0, "input_dim must be divisible by nheads"
        self.head_dim = input_dim // nheads  # Dimension of each head
        
        # Linear projections for Q, K, V (for all heads)
        self.w_q = nn.Linear(input_dim, input_dim, bias=bias)
        self.w_k = nn.Linear(input_dim, input_dim, bias=bias)
        self.w_v = nn.Linear(input_dim, input_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(input_dim, input_dim, bias=bias)
        
    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        """
        This forward pass definition allows for cross-attention. If the input is only X, then it becomes self-attention.

        Args:
            query, key, value: Input tensors of shape [batch_size, seq_len, input_dim]
            attn_mask: Optional mask tensor of shape [seq_len, seq_len] or [batch_size, nhead, seq_len, seq_len]
            is_causal: If True, applies a causal mask automatically (overrides attn_mask if provided)
        
        Returns:
            output: Contextualized output tensor [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = query.shape
        
        # 1. Project inputs to Q, K, V
        Q = self.w_q(query)  # [batch_size, seq_len, input_dim]
        K = self.w_k(key)    # [batch_size, seq_len, input_dim]
        V = self.w_v(value)  # [batch_size, seq_len, input_dim]
        
        # 2. Reshape for multi-head: [batch_size, seq_len, nhead, d_k]
        Q = Q.view(batch_size, seq_len, self.nheads, self.head_dim)
        K = K.view(batch_size, seq_len, self.nheads, self.head_dim)
        V = V.view(batch_size, seq_len, self.nheads, self.head_dim)
        
        # 3. Transpose to [batch_size, nhead, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 4. Compute scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_attention if self.training else 0.0,
            is_causal=is_causal
        )

        # 5. Transpose back: [batch_size, seq_len, nhead, d_k]
        attn_output = attn_output.transpose(1, 2)
        
        # 6. Concatenate heads: [batch_size, seq_len, d_model]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.input_dim)
        
        # 7. Apply output projection
        # This layer is needed to mix the multiple heads outputs
        output = self.out_proj(attn_output)
        
        return output

    def get_attention_weights(self, query, key, attn_mask=None):
        """
        This method allows to extract the attention weights from the layer.
        attn_weights shape: [batch_size, nhead, seq_len, seq_len]
        attn_weights[0, 0] would be a [seq_len, seq_len] matrix,
        showing the attention pattern for the FIRST head for the FIRST input X.
        The element at position [i, j] answers: "For the token at position i (Query), 
        how much did it pay attention to the token at position j (Key)?

        Args:
            query, key, value: Input tensors of shape [batch_size, seq_len, input_dim]
            attn_mask: Optional mask tensor of shape [seq_len, seq_len] or [batch_size, nhead, seq_len, seq_len]
            is_causal: If True, applies a causal mask automatically (overrides attn_mask if provided)
        
        Returns:
            attn_weights: Attention weights tensor [batch_size, nhead, seq_len, seq_len]
        """
        with torch.no_grad():
            batch_size, seq_len, _ = query.shape
            
            # 1. Project inputs to Q, K, V
            Q = self.w_q(query)  # [batch_size, seq_len, input_dim]
            K = self.w_k(key)    # [batch_size, seq_len, input_dim]
            
            # 2. Reshape for multi-head: [batch_size, seq_len, nhead, d_k]
            Q = Q.view(batch_size, seq_len, self.nheads, self.head_dim)
            K = K.view(batch_size, seq_len, self.nheads, self.head_dim)
            
            # 3. Transpose to [batch_size, nhead, seq_len, head_dim]
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)

            # compute the weights
            L, S = Q.size(-2), K.size(-2)
            scale = Q.size(-1) ** -0.5
            attn_bias = torch.zeros(L, S, dtype=Q.dtype, device=Q.device)
            if attn_mask is not None:
                attn_bias = attn_bias + attn_mask
            attn_weights = (Q @ K.transpose(-2, -1) * scale + attn_bias).softmax(dim=-1)
    
        return attn_weights


class MultiHeadSelfAttentionWithWeights(nn.Module):
    """
    Simplified wrapper for self-attention only.
    """
    def __init__(self, input_dim, nheads, dropout_attention=0.1, bias=True):
        super().__init__()
        self.mha = MultiHeadCrossAttentionWithWeights(input_dim, nheads, dropout_attention, bias)
    
    def forward(self, x, attn_mask=None, is_causal=False):
        # For self-attention: use x for query, key, and value
        return self.mha(x, x, x, attn_mask, is_causal)


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
