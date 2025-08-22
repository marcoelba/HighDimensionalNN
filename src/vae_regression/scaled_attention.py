# Scaled attention layer details
import numpy as np
import random
import scipy


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
attn_weights = torch.nn.functional.softmax(torch.from_numpy(attn_scores * scale), dim=-1)  # [batch, nhead, T, T]attn_scores = np.softmax(scores / scale)
attn_weights = attn_weights.numpy()
attn_weights.cumsum(axis=1)

output = attn_weights @ value  # Weighted sum of values [batch, nhead, T, d_k]
