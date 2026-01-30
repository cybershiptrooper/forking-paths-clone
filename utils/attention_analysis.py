"""Utilities for analyzing attention patterns from transformer models."""

from typing import Dict, List, Tuple
import torch
from scipy import stats

from utils.utils import Sentence


def aggregate_attention_by_sentences(
    attention_matrix: torch.Tensor,
    sentences: List[Sentence],
    aggregation: str = 'mean'
) -> torch.Tensor:
    """
    Aggregate token-level attention to sentence-level matrix.
    
    For each pair of sentences (i, j), aggregates attention scores from all
    tokens in sentence i attending to all tokens in sentence j.
    
    Args:
        attention_matrix: Token-level attention matrix with shape
            (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        sentences: List of Sentence namedtuples with (start, end) indices
        aggregation: Aggregation method - 'mean', 'median', or 'max'
        
    Returns:
        Sentence-level attention matrix with shape:
        - (num_sentences, num_sentences) if input was 2D
        - (num_heads, num_sentences, num_sentences) if input was 3D
    """
    has_head_dim = attention_matrix.dim() == 3
    if not has_head_dim:
        attention_matrix = attention_matrix.unsqueeze(0)  # (1, seq_len, seq_len)
    
    num_heads = attention_matrix.shape[0]
    num_sentences = len(sentences)
    
    # Initialize output tensor
    sentence_attention = torch.zeros(num_heads, num_sentences, num_sentences)
    
    for i, sent_i in enumerate(sentences):
        for j, sent_j in enumerate(sentences):
            # Extract attention block from tokens in sent_i to tokens in sent_j
            # sent_i is the "from" sentence (rows), sent_j is the "to" sentence (cols)
            block = attention_matrix[
                :,
                sent_i.start:sent_i.end + 1,
                sent_j.start:sent_j.end + 1
            ]  # (num_heads, len_i, len_j)
            
            # Aggregate over tokens
            if aggregation == 'mean':
                agg_value = block.mean(dim=(1, 2))  # (num_heads,)
            elif aggregation == 'median':
                # Flatten and compute median
                flat = block.flatten(start_dim=1)  # (num_heads, len_i * len_j)
                agg_value = flat.median(dim=1).values
            elif aggregation == 'max':
                agg_value = block.amax(dim=(1, 2))  # (num_heads,)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}. Use 'mean', 'median', or 'max'.")
            
            sentence_attention[:, i, j] = agg_value
    
    if not has_head_dim:
        sentence_attention = sentence_attention.squeeze(0)
    
    return sentence_attention


def compute_vertical_scores(attention_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute vertical attention scores (downstream attention to each position).
    
    "Vertical score" for position j = mean of column j below the diagonal,
    i.e., how much downstream positions (i > j) attend to position j.
    
    For attention matrix A where A[i, j] = attention from position i to position j:
    vertical_score[j] = mean(A[i, j] for all i > j)
    
    Args:
        attention_matrix: Attention matrix with shape (seq_len, seq_len) or 
            (num_heads, seq_len, seq_len)
            
    Returns:
        Vertical scores with shape (seq_len,) or (num_heads, seq_len)
    """
    has_head_dim = attention_matrix.dim() == 3
    if not has_head_dim:
        attention_matrix = attention_matrix.unsqueeze(0)  # (1, seq_len, seq_len)
    
    num_heads, seq_len, _ = attention_matrix.shape
    vertical_scores = torch.zeros(num_heads, seq_len)
    
    for j in range(seq_len - 1):  # Last position has no downstream positions
        # Get column j, but only rows below diagonal (i > j)
        col_below_diag = attention_matrix[:, j + 1:, j]  # (num_heads, seq_len - j - 1)
        if col_below_diag.shape[1] > 0:
            vertical_scores[:, j] = col_below_diag.mean(dim=1)
    
    if not has_head_dim:
        vertical_scores = vertical_scores.squeeze(0)
    
    return vertical_scores


def compute_kurtosis_per_head(vertical_scores: torch.Tensor) -> torch.Tensor:
    """
    Compute kurtosis for each head's vertical score distribution.
    
    Higher kurtosis indicates more "spiky" distributions where attention
    concentrates on specific positions.
    
    Args:
        vertical_scores: Tensor of shape (num_heads, seq_len)
        
    Returns:
        Tensor of kurtosis values with shape (num_heads,)
    """
    if vertical_scores.dim() == 1:
        vertical_scores = vertical_scores.unsqueeze(0)
    
    num_heads = vertical_scores.shape[0]
    kurtosis_values = torch.zeros(num_heads)
    
    for h in range(num_heads):
        scores = vertical_scores[h].numpy()
        # scipy.stats.kurtosis uses Fisher's definition (excess kurtosis)
        # Normal distribution has kurtosis = 0
        kurtosis_values[h] = stats.kurtosis(scores, fisher=True)
    
    return kurtosis_values


def get_top_kurtosis_heads(
    vertical_scores: torch.Tensor,
    k: int = 5
) -> List[Tuple[int, float]]:
    """
    Get top k heads with highest kurtosis (most spiky attention distributions).
    
    Args:
        vertical_scores: Tensor of shape (num_heads, seq_len)
        k: Number of top heads to return
        
    Returns:
        List of (head_idx, kurtosis_value) tuples sorted by kurtosis descending
    """
    kurtosis_values = compute_kurtosis_per_head(vertical_scores)
    
    # Get top k indices
    k = min(k, len(kurtosis_values))
    top_indices = torch.argsort(kurtosis_values, descending=True)[:k]
    
    result = [(int(idx), float(kurtosis_values[idx])) for idx in top_indices]
    return result


def get_top_k_sentences_per_head(
    vertical_scores: torch.Tensor,
    sentences: List[Sentence],
    k: int = 5
) -> Dict[int, List[Tuple[int, float, Sentence]]]:
    """
    Get top k sentences with highest vertical scores for each head.
    
    Args:
        vertical_scores: Tensor of shape (num_heads, num_sentences)
        sentences: List of Sentence namedtuples
        k: Number of top sentences to return per head
        
    Returns:
        Dict mapping head_idx -> [(sentence_idx, score, Sentence), ...]
        sorted by score descending
    """
    if vertical_scores.dim() == 1:
        vertical_scores = vertical_scores.unsqueeze(0)
    
    num_heads = vertical_scores.shape[0]
    result = {}
    
    for h in range(num_heads):
        scores = vertical_scores[h]
        k_actual = min(k, len(scores))
        top_indices = torch.argsort(scores, descending=True)[:k_actual]
        
        head_result = []
        for idx in top_indices:
            idx = int(idx)
            score = float(scores[idx])
            sentence = sentences[idx]
            head_result.append((idx, score, sentence))
        
        result[h] = head_result
    
    return result
