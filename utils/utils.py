import gc
import random
import numpy as np
import torch
from collections import namedtuple
from transformers import PreTrainedModel

MODEL_METADATA = {
    # Deepseek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "nickname" : "deepseek_1b",
        "reasoning" : True,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "nickname": "deepseek_llama_8b",
        "reasoning": True
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "nickname": "deepseek_qwen_7b",
        "reasoning": True
    },
    # Llama
    "meta-llama/Llama-3.2-3B-Instruct": {
        "nickname" : "llama_3b",
        "reasoning" : False,
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "nickname" : "llama_1b",
        "reasoning" : False,
    },
    # Qwen 3
    "Qwen/Qwen3-8B": {
        "nickname": "qwen3_8b",
        "reasoning": True,
    },
    "Qwen/Qwen3-4B": {
        "nickname": "qwen3_4b",
        "reasoning": True,
    },
    "Qwen/Qwen3-14B": {
        "nickname": "qwen3_14b",
        "reasoning": True,
    },
    "Qwen/Qwen3-32B": {
        "nickname": "qwen3_32b",
        "reasoning": True,
    },
    "openai/gpt-oss-120b": {
        "nickname": "gpt_oss_120b",
        "reasoning": True,
    },
    # Gemma 3 (instruction-tuned, no thinking mode: gets the
    # "Let's think step by step." assistant prefill like the other
    # non-reasoning models)
    "google/gemma-3-12b-it": {
        "nickname": "gemma3_12b",
        "reasoning": False,
    },
    # Other
    "microsoft/Phi-4-mini-reasoning": {
        "nickname" : "phi_4b",
        "reasoning" : True,
    },
    "HuggingFaceTB/SmolLM2-360M-Instruct": {
        "nickname": "smol_360m",
        "reasoning": False
    }
}

# should we do something smarter? what about "0.8"?
SENTENCE_DELIMITERS = [
    ".",
    "!",
    "?",
    "\n",
]

# Sentence = (start_token_idx, end_token_idx)
Sentence = namedtuple("Sentence", ["start", "end"])



def set_seed(seed : int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clear_cuda():
    """Clear memory from CUDA - assuming all relevant object references are deleted."""
    # https://discuss.pytorch.org/t/48879/27
    gc.collect()
    torch.cuda.empty_cache()


# Architecture-specific attention module paths
# Easy to extend for new model architectures
ATTENTION_MODULE_PATHS = {
    "llama": lambda model, layer: model.model.layers[layer].self_attn,
    "qwen2": lambda model, layer: model.model.layers[layer].self_attn,
    "qwen": lambda model, layer: model.model.layers[layer].self_attn,
    "qwen3": lambda model, layer: model.model.layers[layer].self_attn,
    "gpt_oss": lambda model, layer: model.model.layers[layer].self_attn,
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
