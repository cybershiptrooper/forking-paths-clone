import torch
import torch.nn as nn
import math
import types
from typing import List, Union, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from utils.utils import Sentence


def load_custom_model_eager(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    device: str = "cuda",
):
    """
    Loads the model with `attn_implementation="eager"`.
    This is CRITICAL: Flash Attention does not expose the attention weights matrix,
    so we must force the explicit (eager) implementation to modify it.
    """
    print(f"Loading {model_name} with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # Force explicit attention matrix calculation
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# --- The Custom Forward Logic ---
def llama_attention_forward_with_ablation(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    A copy of LlamaAttention.forward from Hugging Face (Llama 3.1) with
    ABLATION LOGIC injected after Softmax.
    """
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.config.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.config.num_attention_heads * self.head_dim)
            // self.config.pretraining_tp,
            dim=0,
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            nn.functional.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            nn.functional.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            nn.functional.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.config.num_attention_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    # Apply RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle KV Cache
    use_cache = past_key_values is not None
    if use_cache and past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # Repeat KV for GQA
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # --- Compute Attention Weights ---
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attention_mask is not None:
        causal_mask = attention_mask
        if attention_mask.size() != (bsz, 1, q_len, key_states.shape[-2]):
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Upcast to fp32 for Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # =========================================================================
    # [INJECTION] ABLATION HOOK LOGIC
    # =========================================================================
    if hasattr(self, "_ablation_sentences") and self._ablation_sentences:
        # Clone to avoid modifying gradients/inputs if not intended (though here we modify flow)
        # attn_weights shape: (bsz, num_heads, q_len, k_len)
        _, _, _, k_len = attn_weights.shape

        for sent in self._ablation_sentences:
            start = max(0, sent.start)
            end = min(k_len, sent.end + 1)  # Python slicing is exclusive, so +1
            if start < end:
                # Zero out probability to keys in [start, end)
                # We use masked_fill or direct assignment
                attn_weights[..., :, start:end] = 0.0

        # Renormalize
        row_sums = attn_weights.sum(dim=-1, keepdim=True)
        row_sums = row_sums + 1e-12
        attn_weights = attn_weights / row_sums
    # =========================================================================

    attn_weights = nn.functional.dropout(
        attn_weights, p=self.attention_dropout, training=self.training
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
    else:
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, self.config.hidden_size)
        )

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.config.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.config.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                nn.functional.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, past_key_values


# --- Hook Registration (Patching) Function ---
class AblationHandle:
    def __init__(self, layer, original_method):
        self.layer = layer
        self.original_method = original_method

    def remove(self):
        """Restores the original forward method."""
        self.layer.forward = self.original_method
        if hasattr(self.layer, "_ablation_sentences"):
            del self.layer._ablation_sentences


def ablate_sentences(
    model: nn.Module,
    sentences_to_ablate: List[Sentence],
    layers: Union[int, List[int], str] = "all",
):
    """
    Monkey-patches the attention modules of the model to apply zero-ablation.

    Args:
        model: A HuggingFace model (AutoModelForCausalLM)
        sentences_to_ablate: List of Sentence(start, end) specifying spans
        layers: Which layers to apply ablation to:
            - "all": apply to all attention layers
            - int: single layer index
            - List[int]: specific layer indices

    Returns:
        A list of AblationHandle objects. Call `.remove()` on each to restore.
    """
    # 1. Resolve layers
    if hasattr(model, "model"):
        # Support generic HF Llama structure
        num_layers = len(model.model.layers)
        layer_list = model.model.layers
    else:
        # Fallback if structure is different
        num_layers = len(model.layers)
        layer_list = model.layers

    if layers == "all":
        target_layers = list(range(num_layers))
    elif isinstance(layers, int):
        target_layers = [layers]
    else:
        target_layers = layers

    handles = []

    # 2. Patch each requested layer
    for i in target_layers:
        attn_module = layer_list[i].self_attn

        # Save original method
        original_forward = attn_module.forward

        # Attach the sentences data to the module so the custom function can access it
        attn_module._ablation_sentences = sentences_to_ablate

        # Bind the custom function to this instance
        # MethodType creates a bound method where 'self' is the attn_module
        attn_module.forward = types.MethodType(
            llama_attention_forward_with_ablation, attn_module
        )

        handles.append(AblationHandle(attn_module, original_forward))

    return handles
