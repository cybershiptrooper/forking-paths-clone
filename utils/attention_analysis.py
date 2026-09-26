"""Utilities for analyzing attention patterns from transformer models."""

from typing import Dict, List, Tuple, Optional
import torch
from scipy import stats

from utils.utils import Sentence


def zero_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """
    Zero out the diagonal of an attention matrix.
    
    Useful for excluding self-attention (sentence attending to itself) from analysis.
    
    Args:
        matrix: Attention matrix with shape (seq_len, seq_len) or 
            (num_heads, seq_len, seq_len)
            
    Returns:
        Matrix with diagonal set to zero (same shape as input)
    """
    result = matrix.clone()
    has_head_dim = result.dim() == 3

    if has_head_dim:
        # (num_heads, seq_len, seq_len)
        num_heads, seq_len, _ = result.shape
        for h in range(num_heads):
            result[h].fill_diagonal_(0)
    else:
        # (seq_len, seq_len)
        result.fill_diagonal_(0)

    return result


def apply_gap_filter(matrix: torch.Tensor, gap: int) -> torch.Tensor:
    """
    Filter attention matrix to only include pairs that are at least 'gap' positions apart.

    Only considers attention from position i to position j where |i - j| >= gap.
    This is useful for analyzing long-range attention patterns.

    Args:
        matrix: Attention matrix with shape (seq_len, seq_len) or
            (num_heads, seq_len, seq_len)
        gap: Minimum distance between positions (e.g., gap=4 means only pairs 4+ apart)

    Returns:
        Matrix with entries zeroed out where |i - j| < gap (same shape as input)
    """
    result = matrix.clone()
    has_head_dim = result.dim() == 3

    if has_head_dim:
        num_heads, seq_len, _ = result.shape
    else:
        seq_len = result.shape[0]
        result = result.unsqueeze(0)  # Add head dim temporarily
        num_heads = 1

    # Create mask: True where |i - j| >= gap, False otherwise
    i_indices = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    j_indices = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
    gap_mask = torch.abs(i_indices - j_indices) >= gap  # (seq_len, seq_len)

    # Apply mask to each head
    for h in range(num_heads):
        result[h] = result[h] * gap_mask

    if not has_head_dim:
        result = result.squeeze(0)

    return result


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


def select_top_sentences_by_mean_score(
    top_sentences_per_head: Dict[str, List[Dict]],
    k: int = 10,
    max_token_idx: Optional[int] = None,
    clip_to_max: bool = True,
) -> List[Sentence]:
    """
    Select top K sentences by mean score across all heads.

    Aggregates scores across all heads for each unique sentence (identified by
    token_start and token_end) and returns the top K sentences with highest
    mean scores.

    Args:
        top_sentences_per_head: Dictionary mapping head indices (as strings) to
            lists of sentence dicts. Each dict has keys: sentence_idx, score,
            token_start, token_end.
        k: Number of top sentences to return (default: 10)
        max_token_idx: If provided, only include sentences that start before or at
            this token index. Sentences extending beyond this point will be clipped
            if clip_to_max is True, or excluded if clip_to_max is False.
        clip_to_max: If True and max_token_idx is provided, clip sentences that
            extend beyond max_token_idx. If False, exclude such sentences entirely.
            Default: True

    Returns:
        List of Sentence objects sorted by mean score descending
    """
    # Aggregate scores by sentence (identified by token_start, token_end)
    sentence_scores = {}  # (token_start, token_end) -> list of scores

    for head_idx, sentences_list in top_sentences_per_head.items():
        for sent_dict in sentences_list:
            token_start = sent_dict["token_start"]
            token_end = sent_dict["token_end"]
            score = sent_dict["score"]

            # Filter/clip based on max_token_idx if provided
            if max_token_idx is not None:
                # Skip sentences that start after max_token_idx
                if token_start > max_token_idx:
                    continue
                # Clip sentences that extend beyond max_token_idx
                if clip_to_max and token_end > max_token_idx:
                    token_end = max_token_idx

            key = (token_start, token_end)
            if key not in sentence_scores:
                sentence_scores[key] = []
            sentence_scores[key].append(score)

    # Compute mean score for each sentence
    sentence_means = []
    for (token_start, token_end), scores in sentence_scores.items():
        mean_score = sum(scores) / len(scores)
        sentence_means.append((mean_score, token_start, token_end))

    # Sort by mean score descending and take top k
    sentence_means.sort(reverse=True, key=lambda x: x[0])
    k_actual = min(k, len(sentence_means))

    # Convert to Sentence objects
    result = []
    for mean_score, token_start, token_end in sentence_means[:k_actual]:
        result.append(Sentence(start=token_start, end=token_end))

    return result
