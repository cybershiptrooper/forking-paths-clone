from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_outcome_probability(df, figsize=(10, 6)):
    """
    Plot stacked area chart of outcome_probability at each timestep.
    
    Args:
        df: DataFrame with columns 't', 'outcome', 'outcome_probability'
        figsize: tuple for figure size
    """
    timesteps = sorted(df['t'].unique())
    outcomes = sorted(df['outcome'].unique())
    
    # Build fractions array for each outcome
    fractions = []
    for outcome in outcomes:
        outcome_probs = []
        for t in timesteps:
            row = df[(df['t'] == t) & (df['outcome'] == outcome)]
            if len(row) > 0:
                assert len(row) == 1
                outcome_probs.append(row['outcome_probability'].values[0])
            else:
                outcome_probs.append(0)
        fractions.append(outcome_probs)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(timesteps, fractions, labels=outcomes, alpha=0.8)
    
    # ax.set_xscale('log')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Outcome Probability')
    ax.set_title('Outcome Probability Distribution Over Time')
    ax.legend(loc='upper right', title='Outcome')
    
    plt.tight_layout()
    return fig, ax

def plot_entropy(df, figsize=(10, 6)):
    """
    Plot entropy of the outcome_probability at each timestep.
    
    Args:
        df: DataFrame with columns 't', 'outcome', 'outcome_probability'
        figsize: tuple for figure size
    """
    def entropy_of_distribution(probs: np.ndarray) -> float:
        return -np.sum(probs * np.log(probs + 1e-6))

    timesteps = sorted(df['t'].unique())
    outcomes = sorted(df['outcome'].unique())
    
    # Build entropy array for each outcome
    entropy = []
    for t in timesteps:
        outcome_probs = []
        for outcome in outcomes:
            row = df[(df['t'] == t) & (df['outcome'] == outcome)]
            if len(row) > 0:
                outcome_probs.append(row['outcome_probability'].values[0])
            else:
                outcome_probs.append(0)
        entropy.append(entropy_of_distribution(np.array(outcome_probs)))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(timesteps, entropy)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy of Outcome Probability Distribution Over Time')
    plt.tight_layout()
    return fig, ax


def plot_vertical_attention_scores(
    vertical_scores: torch.Tensor,
    head_to_highlight: Optional[int] = None,
    layer: int = 0,
    convergence_sentence_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot vertical attention scores for all heads on one figure.
    
    Args:
        vertical_scores: Tensor of shape (num_heads, seq_len) containing vertical scores
        head_to_highlight: Optional head index to highlight with thicker line
        layer: Layer index for title
        convergence_sentence_idx: If provided, draw red vertical line at this position
        figsize: Figure size tuple
        
    Returns:
        (fig, ax) tuple
    """
    if vertical_scores.dim() == 1:
        vertical_scores = vertical_scores.unsqueeze(0)
    
    vertical_scores = vertical_scores.numpy()
    num_heads, seq_len = vertical_scores.shape
    positions = np.arange(seq_len)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each head with different colors
    colors = plt.cm.tab20(np.linspace(0, 1, num_heads))
    
    for h in range(num_heads):
        linewidth = 2.5 if h == head_to_highlight else 0.8
        alpha = 1.0 if h == head_to_highlight else 0.6
        label = f"Head {h}" if h == head_to_highlight else None
        ax.plot(positions, vertical_scores[h], color=colors[h], 
                linewidth=linewidth, alpha=alpha, label=label)
    
    # Draw convergence line if provided
    if convergence_sentence_idx is not None:
        ax.axvline(x=convergence_sentence_idx, color='red', linewidth=2, 
                   linestyle='--', label=f'Convergence (pos {convergence_sentence_idx})')
    
    ax.set_xlabel('Sentence Position')
    ax.set_ylabel('Vertical Attention Score')
    ax.set_title(f'Layer {layer} Attention Heads - Vertical Scores')
    
    # Only show legend if there's something to highlight
    if head_to_highlight is not None or convergence_sentence_idx is not None:
        ax.legend(loc='upper right')
    
    ax.set_xlim(0, seq_len - 1)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
    
    plt.tight_layout()
    return fig, ax


def plot_attention_matrix(
    attention_matrix: torch.Tensor,
    head_idx: int,
    layer: int = 0,
    convergence_sentence_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a single attention head as a heatmap.
    
    Args:
        attention_matrix: Attention matrix of shape (seq_len, seq_len)
        head_idx: Head index for title
        layer: Layer index for title
        convergence_sentence_idx: If provided, draw red lines at this position
        figsize: Figure size tuple
        
    Returns:
        (fig, ax) tuple
    """
    if attention_matrix.dim() > 2:
        raise ValueError(f"Expected 2D matrix, got shape {attention_matrix.shape}")
    
    attention_matrix = attention_matrix.numpy()
    seq_len = attention_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto', origin='upper')
    
    # Draw convergence lines if provided
    if convergence_sentence_idx is not None:
        # Horizontal line (attention FROM this sentence)
        ax.axhline(y=convergence_sentence_idx, color='red', linewidth=2, linestyle='-')
        # Vertical line (attention TO this sentence)
        ax.axvline(x=convergence_sentence_idx, color='red', linewidth=2, linestyle='-')
    
    ax.set_xlabel('Key Position (attended to)')
    ax.set_ylabel('Query Position (attending from)')
    ax.set_title(f'Layer {layer} Head {head_idx} - Attention Matrix')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight')
    
    plt.tight_layout()
    return fig, ax