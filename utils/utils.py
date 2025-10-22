import gc
import random
import numpy as np
import torch

MODEL_METADATA = {
    "meta-llama/Llama-3.2-3B-Instruct": {
        "nickname" : "llama_3b",
        "reasoning" : False,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "nickname" : "deepseek_1b",
        "reasoning" : True,
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "nickname" : "llama_1b",
        "reasoning" : False,
    },
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
    ";",
    "!",
    "?"
]

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
