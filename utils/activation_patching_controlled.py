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
    Patched LlamaAttention forward method with source->target ablation logic.
    Logic:
    1. Determine absolute position of current Query tokens.
    2. Check if current Query positions are in `ablate_from_sentences`.
    3. If yes, mask attention to Key positions defined in `sentences_to_ablate`.
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
    # [INJECTION] SOURCE -> TARGET ABLATION LOGIC
    # =========================================================================

    # 1. Retrieve config
    ablate_from = getattr(self, "_ablate_from_sentences", [])
    ablate_to = getattr(self, "_sentences_to_ablate", [])

    if ablate_from and ablate_to:
        # 2. Determine Absolute Position of the current Queries
        # key_states.shape[-2] represents the total sequence length (past + current)
        total_len = key_states.shape[-2]

        # Use cache_position if available to get actual absolute positions
        # During generation with KV cache, cache_position tells us the absolute positions
        if cache_position is not None:
            # cache_position gives us the absolute positions of current queries
            # Shape: (q_len,) - absolute positions in the sequence
            current_query_positions = cache_position.cpu().tolist()
        else:
            # Fallback: assume contiguous positions starting from total_len - q_len
            current_start_idx = total_len - q_len
            current_query_positions = list(range(current_start_idx, total_len))

        # 3. Check Overlap: Do the current queries fall into any `ablate_from` bucket?
        # We need to find which LOCAL rows (0 to q_len) correspond to the global `ablate_from` ranges.
        rows_to_mask = [] # List of (local_start, local_end)

        for sent in ablate_from:
            # Find which local query indices fall within this sentence range
            matching_indices = []
            for local_idx, abs_pos in enumerate(current_query_positions):
                if sent.start <= abs_pos < sent.end:
                    matching_indices.append(local_idx)

            if matching_indices:
                # Get contiguous ranges of matching indices
                # Sort to ensure we get ranges correctly
                matching_indices.sort()
                local_start = matching_indices[0]
                local_end = matching_indices[-1] + 1
                rows_to_mask.append((local_start, local_end))

        # 4. Apply Masking
        if rows_to_mask:
            modified = False
            _, _, _, k_len = attn_weights.shape

            for (r_start, r_end) in rows_to_mask:
                for target in ablate_to:
                    # Target range (Keys/Columns)
                    c_start = max(0, target.start)
                    c_end = min(k_len, target.end + 1) # Python slicing exclusive

                    if c_start < c_end:
                        # Zero out: specific rows (queries) cannot attend to specific cols (targets)
                        # attn_weights: (bsz, heads, q_len, k_len)
                        attn_weights[..., r_start:r_end, c_start:c_end] = 0.0
                        modified = True

            # 5. Renormalize if necessary
            if modified:
                row_sums = attn_weights.sum(dim=-1, keepdim=True) + 1e-12
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
        if hasattr(self.layer, "_ablate_from_sentences"):
            del self.layer._ablate_from_sentences
        if hasattr(self.layer, "_sentences_to_ablate"):
            del self.layer._sentences_to_ablate


def ablate_sentences(
    model: nn.Module,
    sentences_to_ablate: List[Sentence],  # TARGETS (Keys)
    ablate_from_sentences: List[Sentence],  # SOURCES (Queries)
    layers: Union[int, List[int], str] = "all",
):
    """
    Patches the model to ablate attention.

    Logic:
    When the model is calculating embeddings for tokens in `ablate_from_sentences` (Sources),
    it is forbidden from attending to tokens in `sentences_to_ablate` (Targets).

    Args:
        model: A HuggingFace model (AutoModelForCausalLM)
        sentences_to_ablate: List of Sentence(start, end) specifying target key positions
        ablate_from_sentences: List of Sentence(start, end) specifying source query positions
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

        # Attach configuration to the module
        attn_module._ablate_from_sentences = ablate_from_sentences
        attn_module._sentences_to_ablate = sentences_to_ablate

        # Bind custom method
        attn_module.forward = types.MethodType(
            llama_attention_forward_with_ablation, attn_module
        )

        handles.append(AblationHandle(attn_module, original_forward))

    return handles
