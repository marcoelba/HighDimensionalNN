# Scaled attention layer details
import numpy as np
import random
import scipy
import torch

import os
os.chdir("./src")

from vae_attention.modules.sinusoidal_position_encoder import SinusoidalPositionalEncoding


n = 5
p = 3

x = np.random.randn(n, p)

# generating the weight matrices
np.random.seed(42) # to allow us to reproduce the same attention values
W_Q = np.random.randint(3, size=(p, p))
W_K = np.random.randint(3, size=(p, p))
W_V = np.random.randint(3, size=(p, p))

# generating the queries, keys and values, for each observation
query = x @ W_Q
key = x @ W_K
value = x @ W_V

# scoring the first query vector against all key vectors
attn_scores = query @ key.transpose()

# computing the weights by a softmax operation and scaling
scale = p ** -0.5
attn_weights = torch.nn.functional.softmax(torch.from_numpy(attn_scores * scale), dim=-1)
attn_weights = attn_weights.numpy()
attn_weights.cumsum(axis=1)

output = attn_weights @ value  # Weighted sum of values


# ---------- With time dimension ----------
n = 10
p = 4
T = 5

np.random.seed(35)
x = np.random.randn(n, p)
x = np.repeat(x[:, None, :], T, axis=1)
x.shape
x[0, 0, :]
x[0, 1, :]

pos_encoder = SinusoidalPositionalEncoding(
    p,
    10
)
x = pos_encoder(torch.tensor(x)).numpy()
x.shape
x[0, 0, :]
x[0, 1, :]


# generating the weight matrices
np.random.seed(42)
W_Q = np.random.randn(p, p)
W_K = np.random.randn(p, p)
W_V = np.random.randn(p, p)

# generating the queries, keys and values, for each observation
query = x @ W_Q
key = x @ W_K
value = x @ W_V
print("Shape: ", query.shape)

out = torch.nn.functional.scaled_dot_product_attention(
    torch.tensor(query), torch.tensor(key), torch.tensor(value),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False
)
out.shape

# scoring the first query vector against all key vectors
attn_scores = np.einsum('... i e, ... j e -> ... i j', query, key)  # shape [n, T, T]
attn_scores.shape

# computing the weights by a softmax operation and scaling
scale = p ** -0.5
attn_weights = torch.nn.functional.softmax(torch.from_numpy(attn_scores * scale), dim=-1)  # [batch, nhead, T, T]attn_scores = np.softmax(scores / scale)
attn_weights.shape
attn_weights[0, 0, :]
attn_weights[1, 0, :]
attn_weights = attn_weights.numpy()
attn_weights.cumsum(axis=1)

output = attn_weights @ value  # Weighted sum of values [batch, nhead, T, d_k]
output.shape
output[0, 0, :]
output[0, 1, :]
