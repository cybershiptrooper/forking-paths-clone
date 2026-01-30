import torch
from collections import namedtuple
from typing import List, Union
from nnsight import LanguageModel

# Sentence = (start_token_idx, end_token_idx)
# Note: 'end' is exclusive in Python slicing, so we assume [start, end)
Sentence = namedtuple("Sentence", ["start", "end"])

def get_model(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda"):
    """
    Loads the model with 'eager' attention implementation.
    This is CRITICAL: Flash/SDPA attention kernels often fuse the mask 
    and computation, preventing intervention. 'eager' exposes the mask.
    """
    model = LanguageModel(
        model_name, 
        device_map=device_map, 
        attn_implementation="eager" 
    )
    return model

def ablate_sentences(model: LanguageModel, 
                     sentences_to_ablate: List[Sentence], 
                     layers: Union[int, List[int], str] = "all",
                     prompt: str = None,
                     max_new_tokens: int = 10):
    """
    Ablates attention to specific sentences by modifying the attention mask 
    before it enters the attention mechanism.
    
    This preserves KV Caching, making it O(N) instead of O(N^2) for generation.
    """
    
    # 1. Parse Layers
    if layers == "all":
        # DeepSeek/Llama access pattern for layers
        layers_to_ablate = range(len(model.model.layers))
    elif isinstance(layers, int):
        layers_to_ablate = [layers]
    else:
        layers_to_ablate = layers

    # 2. Context Manager for Generation
    # We use .generate() which handles the KV cache loop efficiently
    with model.generate(prompt, max_new_tokens=max_new_tokens) as generator:
        
        # 3. Apply Hooks
        for layer_idx in layers_to_ablate:
            
            # We hook the 'self_attn' module inputs.
            # LlamaAttention.forward signature: 
            # (hidden_states, attention_mask, position_ids, past_key_value, ...)
            # input[1] is the attention_mask.
            attn_module = model.model.layers[layer_idx].self_attn
            
            # Access the mask proxy
            # Shape is typically (Batch, 1, Query_Len, Key_Len)
            current_mask = attn_module.inputs[1]
            
            # Iterate over sentences and apply ablation
            for sentence in sentences_to_ablate:
                # We want to mask out the columns (Keys) corresponding to the sentence
                # [:, :, :, start:end] covers all batches, all heads, all queries, specific keys
                
                # Check bounds to avoid errors (clipping to context length handled by slice mostly)
                s_start = max(0, sentence.start)
                s_end = sentence.end
                
                # Apply -inf (effectively 0 after softmax)
                # We use -1e9 for float stability, or -inf if dtype permits
                current_mask[:, :, :, s_start:s_end] = -torch.finfo(torch.float32).max

    return model