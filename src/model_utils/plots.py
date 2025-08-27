import matplotlib.pyplot as plt


def plot_attention_weights(attn_weights, observation, layer_name="Attention"):
    """
    Plot attention weights for all heads in a grid.
    
    Args:
        attn_weights: Tensor of shape [batch_size, nhead, seq_len, seq_len]
        observation: Observation to plot
        layer_name: Name for the plot title
    """
    batch_size, nhead, seq_len, _ = attn_weights.shape
    
    # Use weights from first batch element
    weights = attn_weights[observation].detach().cpu().numpy()
    
    # Create subplot grid
    fig, axes = plt.subplots(1, nhead, figsize=(4 * nhead, 4))
    if nhead == 1:
        axes = [axes]  # Make it iterable
    
    for i, ax in enumerate(axes):
        im = ax.imshow(weights[i], cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Head {i+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'{layer_name} Weights (Batch 0)')
    plt.tight_layout()
    plt.show()
