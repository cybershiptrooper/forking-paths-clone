"""Abstract base class for circuit discovery algorithms."""

import types
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Union

import torch

from utils.masks import NodeMask
from utils.utils import Sentence, get_attention_module


class AblationHandle:
    """Manages the lifecycle of a monkey-patched attention forward."""

    def __init__(self, module, original_forward):
        self.module = module
        self.original_forward = original_forward

    def remove(self):
        """Restore original forward and clean up attributes."""
        self.module.forward = self.original_forward
        for attr in [
            "_circuit_mask",
            "_token_to_sent",
            "_gap_filter",
            "_renormalize_masked_attn",
        ]:
            if hasattr(self.module, attr):
                delattr(self.module, attr)


class CircuitDiscovery(ABC):
    """Base class for circuit discovery algorithms.

    Subclasses implement specific algorithms (nodewise attribution,
    subnetwork probing, EAP) that learn which sentence-to-sentence
    attention edges are important for the model's output.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        layers: List[int],
        objective_fn: Callable,
        sentence_gap: int = 1,
        ablate_non_target_layers: bool = False,
        renormalize_masked_attention: bool = True,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.objective_fn = objective_fn
        self.sentence_gap = sentence_gap
        self.ablate_non_target_layers = ablate_non_target_layers
        self.renormalize_masked_attention = renormalize_masked_attention

    def _build_token_to_sentence_map(
        self, sentences: List[Sentence], seq_len: int
    ) -> torch.Tensor:
        """Map each token position to its sentence index.

        Tokens not in any sentence get -1 (will map to sentinel in mask expansion).
        """
        mapping = torch.full((seq_len,), -1, dtype=torch.long)
        for idx, sent in enumerate(sentences):
            mapping[sent.start : sent.end + 1] = idx
        return mapping

    def _patch_model(
        self,
        masks: dict,
        token_to_sent: torch.Tensor,
        gap_filter: torch.Tensor,
        custom_forward_fn,
    ) -> List[AblationHandle]:
        """Patch model layers with mask-aware attention forward.

        Args:
            masks: Dict mapping layer_idx -> mask tensor (num_heads, num_sents, num_sents)
            token_to_sent: (seq_len,) mapping token -> sentence index
            gap_filter: (num_sents, num_sents) boolean gap filter
            custom_forward_fn: The custom forward function to bind

        Returns:
            List of AblationHandle for cleanup.
        """
        handles = []
        for layer_idx in self.layers:
            attn_module = get_attention_module(self.model, layer_idx)
            original_forward = attn_module.forward

            attn_module._circuit_mask = masks.get(layer_idx)
            attn_module._token_to_sent = token_to_sent
            attn_module._gap_filter = gap_filter
            attn_module._renormalize_masked_attn = self.renormalize_masked_attention

            attn_module.forward = types.MethodType(custom_forward_fn, attn_module)
            handles.append(AblationHandle(attn_module, original_forward))

        return handles

    def _unpatch_model(self, handles: List[AblationHandle]):
        """Restore original forwards."""
        for h in handles:
            h.remove()

    def _patch_non_target_layers(
        self,
        num_heads: int,
        num_sents: int,
        token_to_sent: torch.Tensor,
        gap_filter: torch.Tensor,
        custom_forward_fn,
    ) -> List[AblationHandle]:
        """Patch all layers NOT in self.layers with zero masks (fully ablated).

        Args:
            num_heads: Number of attention heads per layer
            num_sents: Number of sentences
            token_to_sent: (seq_len,) mapping token -> sentence index
            gap_filter: (num_sents, num_sents) boolean gap filter
            custom_forward_fn: The custom forward function to bind

        Returns:
            List of AblationHandle for cleanup.
        """
        device = next(self.model.parameters()).device
        num_layers = self.model.config.num_hidden_layers
        target_set = set(self.layers)
        non_target = [l for l in range(num_layers) if l not in target_set]

        handles = []
        for layer_idx in non_target:
            attn_module = get_attention_module(self.model, layer_idx)
            original_forward = attn_module.forward

            # Zero mask = fully ablated
            zero_mask = torch.zeros(
                num_heads, num_sents, num_sents, device=device
            )
            attn_module._circuit_mask = zero_mask
            attn_module._token_to_sent = token_to_sent
            attn_module._gap_filter = gap_filter
            attn_module._renormalize_masked_attn = self.renormalize_masked_attention

            attn_module.forward = types.MethodType(custom_forward_fn, attn_module)
            handles.append(AblationHandle(attn_module, original_forward))

        return handles

    @abstractmethod
    def discover(
        self,
        input_ids: torch.Tensor,
        sentences: List[Sentence],
        continuations: List[torch.Tensor],
        mask_mode: str = "prefix",
        num_prefix_sentences: Optional[int] = None,
        **kwargs,
    ) -> NodeMask:
        """Run circuit discovery.

        Args:
            input_ids: (1, prompt_len) tokenized prompt up to analysis timestep
            sentences: List of Sentence(start, end) — prefix + optional generation
            continuations: List of (1, cont_len) token ID tensors for each branch
            mask_mode: "prefix", "generation", or "both"
            num_prefix_sentences: How many sentences are prefix (rest are generation).
                Defaults to len(sentences).

        Returns:
            NodeMask with per-(layer, head, src_sent, tgt_sent) attribution scores.
        """
        ...
