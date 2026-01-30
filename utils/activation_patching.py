from utils.utils import Sentence
from typing import List, Union

import torch
import transformer_lens
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_custom_model_into_transformerlens(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    base_model_name: str = "meta-llama/Llama-3.1-8B",
    device: str = "cuda",
):
    """
    Load the deepseek-ai/DeepSeek-R1-Distill-Llama-8B model
    into a TransformerLens HookedTransformer.

    This loads the HF LLaMA model and then wraps it with TransformerLens,
    enabling internal hooks on attention and other activations.

    Args:
        device: device to load the model on, e.g. "cuda" or "cpu"

    Returns:
        A tuple (hl_model, tokenizer)
        - hl_model: TransformerLens HookedTransformer
        - tokenizer: HF tokenizer for tokenization
    """
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # DeepSeek-R1 is often optimized for bf16
        device_map=device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Initialize TransformerLens
    # We pass "meta-llama/Meta-Llama-3.1-8B" as the model_name so TransformerLens
    # knows to use the Llama-3 specific hook points and folding logic.
    print("Converting to HookedTransformer...")
    tl_model = HookedTransformer.from_pretrained(
        base_model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        fold_ln=False,  # Recommended False when loading custom weights to prevent mismatches
        center_writing_weights=False,
        center_unembed=False,
    )

    return tl_model, tokenizer


def make_ablation_hook(sentences: List[Sentence]):
    def hook_fn(attn_probs, hook):
        """
        Hook for TransformerLens attention probabilities.

        `attn_probs` shape: (batch, heads, seq_len, seq_len)
        Zero out selected columns then renormalize.
        """
        mod = attn_probs.clone()
        _, _, q_len, k_len = mod.shape

        for sent in sentences:
            start = max(0, sent.start)
            end = min(k_len, sent.end + 1)
            if start < end:
                # Zero out probability to keys in [start, end)
                mod[..., :, start:end] = 0.0

        # Renormalize so each query row sums to 1
        row_sums = mod.sum(dim=-1, keepdim=True)
        row_sums = row_sums + 1e-12
        mod = mod / row_sums
        return mod

    return hook_fn


def ablate_sentences(
    model: HookedTransformer,
    sentences_to_ablate: List[Sentence],
    layers: Union[int, List[int], str] = "all",
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register TransformerLens hooks that zero-ablate attention for specific token spans
    in specified layers of a HookedTransformer.

    For each target sentence span, we zero out the attention probabilities
    (post-softmax) to those key positions and renormalize the distributions so
    rows still sum to 1.

    Args:
        model: A HookedTransformer loaded via TransformerLens
        sentences_to_ablate: List of Sentence(start, end) specifying spans
        layers: Which layers to apply ablation to:
            - "all": apply to all attention layers
            - int: single layer index
            - List[int]: specific layer indices

    Returns:
        A list of hook handles. Call `.remove()` to remove each hook.
    """
    # Resolve layers
    num_layers = model.cfg.n_layers
    if layers == "all":
        layers_to_ablate = list(range(num_layers))
    elif isinstance(layers, int):
        layers_to_ablate = [layers]
    else:
        layers_to_ablate = layers

    # Attach hooks per layer hook point
    for layer_idx in layers_to_ablate:
        hook_name = f"blocks.{layer_idx}.attn.hook_pattern"
        model.add_hook(hook_name, make_ablation_hook(sentences_to_ablate))
