"""Utilities for extracting activations and attention patterns from transformer models."""

from typing import List, Optional
import torch
from tqdm import trange
from transformers import PreTrainedModel


# Architecture-specific attention module paths
# Easy to extend for new model architectures
ATTENTION_MODULE_PATHS = {
    "llama": lambda model, layer: model.model.layers[layer].self_attn,
    "qwen2": lambda model, layer: model.model.layers[layer].self_attn,
    "qwen": lambda model, layer: model.model.layers[layer].self_attn,
    # Easy to add new architectures:
    # "phi": lambda model, layer: model.model.layers[layer].mixer,
    # "mistral": lambda model, layer: model.model.layers[layer].self_attn,
}


def get_attention_module(model: PreTrainedModel, layer: int):
    """Auto-detect model architecture and return attention module.
    
    Args:
        model: The pretrained model
        layer: The layer index
        
    Returns:
        The attention module for the specified layer
        
    Raises:
        ValueError: If the model type is not supported
    """
    model_type = model.config.model_type.lower()
    if model_type in ATTENTION_MODULE_PATHS:
        return ATTENTION_MODULE_PATHS[model_type](model, layer)
    raise ValueError(
        f"Unsupported model type: {model_type}. "
        f"Supported types: {list(ATTENTION_MODULE_PATHS.keys())}. "
        f"Add to ATTENTION_MODULE_PATHS in activation_caching.py to support new architectures."
    )


def get_activations(
    model: PreTrainedModel,
    X: dict,
    layer: int,
    batch_size: Optional[int] = None,
    efficient_mode: bool = False
) -> torch.Tensor:
    """
    Extract activations from a specific layer of the model.
    
    Args:
        model: The pretrained model
        X: Dictionary with 'input_ids' and 'attention_mask'
        layer: The layer index to extract activations from
        batch_size: If provided, process in batches
        efficient_mode: If True, use forward hooks to capture only the target layer's 
                       activations (saves GPU memory by not storing all hidden states)
                       
    Returns:
        Tensor of activations with shape (batch_size, seq_len, hidden_dim) or 
        (seq_len, hidden_dim) if batch_size=1 and squeezed
    """
    model.eval()

    if efficient_mode:
        # Use forward hooks to capture only the specified layer's activations
        captured_activations = []
        
        def activation_hook(module, input, output):
            # output is typically (batch_size, seq_len, hidden_dim) or a tuple
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            captured_activations.append(hidden_states.detach().float().cpu())
        
        # Register hook on the target layer
        hook_handle = model.model.layers[layer].register_forward_hook(activation_hook)
        
        try:
            if batch_size is not None:
                activations = []
                for b in trange(0, len(X['input_ids']), batch_size, desc="Collecting activations (efficient)..."):
                    captured_activations.clear()
                    batch_inputs = {
                        'input_ids': X['input_ids'][b:b + batch_size].to(model.device),
                        'attention_mask': X['attention_mask'][b:b + batch_size].to(model.device)
                    }
                    with torch.no_grad():
                        model(**batch_inputs, output_hidden_states=False)
                    activations.append(captured_activations[0])
                return torch.cat(activations, dim=0)
            else:
                with torch.no_grad():
                    model(**X, output_hidden_states=False)
                return captured_activations[0].squeeze()
        finally:
            hook_handle.remove()
    
    # Original method: output_hidden_states=True (stores all layer activations)
    if batch_size is not None:
        activations = []
        for b in trange(0, len(X['input_ids']), batch_size, desc="Collecting activations..."):
            batch_inputs = {
                'input_ids': X['input_ids'][b:b + batch_size].to(model.device),
                'attention_mask': X['attention_mask'][b:b + batch_size].to(model.device)
            }
            with torch.no_grad():
                batch_outputs = model(**batch_inputs, output_hidden_states=True)
                batch_activations = batch_outputs.hidden_states[layer].detach().float().cpu()
                activations.append(batch_activations)
        activations = torch.cat(activations, dim=0)
        return activations

    with torch.no_grad():
        outputs = model(**X, output_hidden_states=True)
    
    activations = outputs.hidden_states[layer].squeeze().float().cpu()

    return activations


def get_attention_patterns(
    model: PreTrainedModel,
    X: dict,
    layer: int,
    heads: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Extract attention patterns from a specific layer of the model.
    
    Args:
        model: The pretrained model
        X: Dictionary with 'input_ids' and optionally 'attention_mask'
        layer: The layer index to extract attention from
        heads: Optional list of head indices to extract. If None, extract all heads.
        
    Returns:
        Tensor of attention patterns with shape:
        - (num_heads, seq_len, seq_len) if heads=None
        - (len(heads), seq_len, seq_len) if heads specified
        
        Attention[h, i, j] = attention from position i to position j for head h
    """
    model.eval()
    
    # Ensure attention_mask is present
    if 'attention_mask' not in X:
        X = {**X, 'attention_mask': torch.ones_like(X['input_ids'])}
    
    with torch.no_grad():
        outputs = model(**X, output_attentions=True)
    
    # Check if attentions were actually returned
    assert outputs.attentions is not None, (
        "Model did not return attention weights. This usually happens when using "
        "Flash Attention. Load the model with attn_implementation='eager' to get attention patterns."
    )
    
    # Check layer index is valid
    num_layers = len(outputs.attentions)
    assert layer < num_layers, (
        f"Layer {layer} out of range. Model has {num_layers} layers (valid indices: 0-{num_layers - 1})."
    )
    
    # outputs.attentions is a tuple of tensors, one per layer
    # Each tensor has shape (batch_size, num_heads, seq_len, seq_len)
    attention = outputs.attentions[layer]  # (batch_size, num_heads, seq_len, seq_len)
    
    # Remove batch dimension (assuming batch_size=1)
    attention = attention.squeeze(0)  # (num_heads, seq_len, seq_len)
    
    # Filter to specific heads if requested
    if heads is not None:
        attention = attention[heads]  # (len(heads), seq_len, seq_len)
    
    return attention.float().cpu()
