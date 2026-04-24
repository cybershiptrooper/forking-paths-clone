"""Circuit discovery algorithms for learning attention masks."""

import torch

# Force CUDA autocast to bf16 process-wide. Every model in this pipeline is
# loaded at bf16; without this line, ``torch.amp.autocast("cuda")`` defaults
# to fp16, which (a) mismatches the weight dtype and (b) silently NaNs when
# the log-mask path feeds very small values through log() — 1e-30 underflows
# to 0 in fp16, log(0) = -inf, and backward gives NaN. bf16 has fp32-like
# exponent range and survives log(1e-30) ≈ -69 without underflow.
torch.set_autocast_dtype("cuda", torch.bfloat16)

from utils.circuit_discovery.base import CircuitDiscovery
from utils.circuit_discovery.common import (
    make_attention_forward,
    apply_sentence_mask,
    expand_sentence_mask_to_tokens,
)
from utils.circuit_discovery.nodewise_attribution import NodewiseAttribution
from utils.circuit_discovery.nodewise_attribution_attention import (
    NodewiseAttribution as NodewiseAttributionAttention,
)
from utils.circuit_discovery.nodewise_activation_patching import (
    NodewiseActivationPatching,
)
from utils.circuit_discovery.factory import create_circuit_discovery

__all__ = [
    "CircuitDiscovery",
    "NodewiseAttribution",
    "NodewiseAttributionAttention",
    "NodewiseActivationPatching",
    "create_circuit_discovery",
    "make_attention_forward",
    "apply_sentence_mask",
    "expand_sentence_mask_to_tokens",
]
