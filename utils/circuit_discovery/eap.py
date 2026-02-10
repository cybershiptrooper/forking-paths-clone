from __future__ import annotations

import math
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from utils.circuit_discovery.base import CircuitDiscoveryAlgorithm
from utils.masks import EdgewiseMask


def llama_attention_forward_with_eap(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Any,
):
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

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attention_mask is not None:
        causal_mask = attention_mask
        if attention_mask.size() != (bsz, 1, q_len, key_states.shape[-2]):
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # =========================================================================
    # [INJECTION] EAP MASK LOGIC
    # =========================================================================
    eap_mask = getattr(self, "_eap_mask", None)
    chunk_map = getattr(self, "_eap_chunk_map", None)
    gap_mask = getattr(self, "_eap_gap_mask", None)
    analysis_timestep = getattr(self, "_eap_analysis_timestep", None)
    layer_index = getattr(self, "_eap_layer_index", None)

    if (
        eap_mask is not None
        and chunk_map is not None
        and analysis_timestep is not None
        and layer_index is not None
    ):
        k_len = attn_weights.shape[-1]
        total_len = key_states.shape[-2]

        if cache_position is not None:
            current_query_positions = cache_position.tolist()
        else:
            current_start_idx = total_len - q_len
            current_query_positions = list(range(current_start_idx, total_len))

        k_prefix_len = min(k_len, analysis_timestep)
        if k_prefix_len > 0:
            k_chunk_ids = chunk_map[:k_prefix_len]
            mask_layer = torch.sigmoid(eap_mask[layer_index])
            if gap_mask is not None:
                mask_layer = mask_layer * gap_mask.unsqueeze(0) + (1.0 - gap_mask).unsqueeze(0)

            for local_q, abs_pos in enumerate(current_query_positions):
                if abs_pos >= analysis_timestep or abs_pos < 0:
                    continue
                q_chunk = int(chunk_map[abs_pos].item())
                mask_for_q = mask_layer[:, q_chunk, :]
                mask_for_keys = mask_for_q[:, k_chunk_ids]
                attn_weights[:, :, local_q, :k_prefix_len] = (
                    attn_weights[:, :, local_q, :k_prefix_len]
                    * mask_for_keys.unsqueeze(0)
                )

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


class EAPHandle:
    def __init__(self, layer: nn.Module, original_method: Any):
        self.layer = layer
        self.original_method = original_method

    def remove(self) -> None:
        self.layer.forward = self.original_method
        for attr in [
            "_eap_mask",
            "_eap_chunk_map",
            "_eap_gap_mask",
            "_eap_analysis_timestep",
            "_eap_layer_index",
        ]:
            if hasattr(self.layer, attr):
                delattr(self.layer, attr)


class EAPDiscovery(CircuitDiscoveryAlgorithm):
    def __init__(
        self,
        model_name: str,
        layers: List[int],
        analysis_timestep: int,
        sentence_chunks: List[Dict[str, int]],
        sentence_gap: int = 1,
        device: str = "cuda",
        mask_init: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.layers = layers
        self.analysis_timestep = analysis_timestep
        self.sentence_chunks = sentence_chunks
        self.sentence_gap = sentence_gap
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_heads = self.model.config.num_attention_heads
        self.num_chunks = len(sentence_chunks)

        init = torch.full(
            (len(layers), self.num_heads, self.num_chunks, self.num_chunks),
            float(mask_init),
            device=self.model.device,
        )
        init = torch.clamp(init, 1e-4, 1 - 1e-4)
        self.mask_logits = nn.Parameter(torch.log(init / (1 - init)))

        self.chunk_map = self._build_chunk_map(sentence_chunks, analysis_timestep).to(
            self.model.device
        )
        self.gap_mask = self._build_gap_mask(self.num_chunks, sentence_gap).to(
            self.model.device
        )

    @staticmethod
    def _build_chunk_map(
        sentence_chunks: List[Dict[str, int]], analysis_timestep: int
    ) -> torch.LongTensor:
        chunk_map = torch.zeros(analysis_timestep, dtype=torch.long)
        for idx, chunk in enumerate(sentence_chunks):
            start = max(0, chunk["start"])
            end = min(analysis_timestep - 1, chunk["end"])
            if start <= end:
                chunk_map[start : end + 1] = idx
        return chunk_map

    @staticmethod
    def _build_gap_mask(num_chunks: int, gap: int) -> torch.Tensor:
        if gap <= 0:
            return torch.ones(num_chunks, num_chunks)
        i_idx = torch.arange(num_chunks).unsqueeze(1)
        j_idx = torch.arange(num_chunks).unsqueeze(0)
        gap_mask = (torch.abs(i_idx - j_idx) >= gap).float()
        return gap_mask

    def _patch_model(self) -> List[EAPHandle]:
        handles: List[EAPHandle] = []
        layer_list = self.model.model.layers
        layer_index_map = {layer: i for i, layer in enumerate(self.layers)}
        for layer_idx in self.layers:
            attn_module = layer_list[layer_idx].self_attn
            original_forward = attn_module.forward
            attn_module._eap_mask = self.mask_logits
            attn_module._eap_chunk_map = self.chunk_map
            attn_module._eap_gap_mask = self.gap_mask
            attn_module._eap_analysis_timestep = self.analysis_timestep
            attn_module._eap_layer_index = layer_index_map[layer_idx]
            attn_module.forward = types.MethodType(
                llama_attention_forward_with_eap, attn_module
            )
            handles.append(EAPHandle(attn_module, original_forward))
        return handles

    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, use_cache=False)
        return outputs.logits

    def learn_mask(
        self,
        prompt: str,
        prompt_token_ids: List[int],
        branch_token_ids: List[List[int]],
        objective_fn,
        num_steps: int = 50,
        lr: float = 1e-1,
    ) -> Tuple[EdgewiseMask, Dict[str, Any]]:
        prompt_len = len(prompt_token_ids)
        baseline_logits: List[torch.Tensor] = []

        with torch.no_grad():
            for branch_ids in branch_token_ids:
                input_ids = torch.tensor(
                    [prompt_token_ids + branch_ids], device=self.model.device
                )
                logits = self._forward_logits(input_ids)
                branch_len = len(branch_ids)
                start = max(0, prompt_len - 1)
                end = start + branch_len
                baseline_logits.append(logits[:, start:end, :].detach().cpu())

        handles = self._patch_model()
        optimizer = torch.optim.Adam([self.mask_logits], lr=lr)
        losses: List[float] = []

        try:
            for step in range(num_steps):
                optimizer.zero_grad()
                total_loss = 0.0
                for i, branch_ids in enumerate(branch_token_ids):
                    input_ids = torch.tensor(
                        [prompt_token_ids + branch_ids], device=self.model.device
                    )
                    logits = self._forward_logits(input_ids)
                    branch_len = len(branch_ids)
                    start = max(0, prompt_len - 1)
                    end = start + branch_len
                    masked_logits = logits[:, start:end, :]
                    baseline = baseline_logits[i].to(masked_logits.device)
                    total_loss = total_loss + objective_fn(baseline, masked_logits)

                loss = total_loss / max(1, len(branch_token_ids))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

        finally:
            for h in handles:
                h.remove()

        with torch.no_grad():
            mask_vals = torch.sigmoid(self.mask_logits)
            if self.gap_mask is not None:
                mask_vals = mask_vals * self.gap_mask.unsqueeze(0).unsqueeze(0) + (
                    1.0 - self.gap_mask
                ).unsqueeze(0).unsqueeze(0)

        edge_mask = EdgewiseMask(
            model_name=self.model_name,
            layers=self.layers,
            num_heads=self.num_heads,
            sentence_chunks=self.sentence_chunks,
            gap=self.sentence_gap,
            analysis_timestep=self.analysis_timestep,
            mask_values=mask_vals.detach().cpu().tolist(),
            prompt=prompt,
            prompt_len=prompt_len,
            metadata={"num_steps": num_steps, "lr": lr},
        )

        metrics = {
            "losses": losses,
            "num_steps": num_steps,
            "lr": lr,
        }
        return edge_mask, metrics
