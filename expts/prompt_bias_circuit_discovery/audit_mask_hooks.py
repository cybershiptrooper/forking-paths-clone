"""CPU checks of the attention helpers on a tiny, random Qwen3.

This checks implementation invariants, not behavior of the trained 8B model.
It also demonstrates why an unassigned token after the prompt is not an
information-free attention sink: it can read and relay the prompt.

Reproduce from the repository root:
    OMP_NUM_THREADS=2 .venv/bin/python -m \
        expts.prompt_bias_circuit_discovery.audit_mask_hooks \
        --output results/prompt_bias_v2/design_audit/mask_hook_sanity.json
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

from utils.circuit_discovery.common import expand_sentence_mask_to_tokens
from utils.circuit_discovery.edits.nodewise_attribution_sdpa import _expand_mask_to_log_additive
from utils.circuit_eval import install_clean_sdpa_forward, install_mask_hooks, remove_handles
from utils.masks import build_causal_filter, build_combined_filter, build_gap_filter, build_mode_filter
from utils.utils import get_attention_module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='results/prompt_bias_v2/design_audit/mask_hook_sanity.json')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config = Qwen3Config(vocab_size=64, hidden_size=48, intermediate_size=96,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=12, max_position_embeddings=64)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).eval()
    original_ids = torch.tensor([[3, 7, 11, 15, 18, 21, 24, 25, 27]])
    layers = list(range(config.num_hidden_layers))
    heads = config.num_attention_heads

    def frozen_mask(n):
        return build_combined_filter(build_gap_filter(n, 1), build_mode_filter(n, n, 'prefix'),
                                     build_causal_filter(n), None)

    def forward(ids):
        with torch.no_grad():
            return model(ids, use_cache=False).logits.detach()

    def masked_forward(ids, sentence_mask, mapping):
        combined = frozen_mask(sentence_mask.shape[-1])
        for i in layers:
            attn = get_attention_module(model, i)
            attn._circuit_mask = sentence_mask[None].expand(heads, -1, -1)
            attn._token_to_sent = mapping
            attn._gap_filter = combined
        try:
            return forward(ids)
        finally:
            for i in layers:
                attn = get_attention_module(model, i)
                for attr in ('_circuit_mask', '_token_to_sent', '_gap_filter'):
                    delattr(attn, attr)

    def max_diff(a, b):
        return float((a - b).abs().max())

    # P tokens [0:3], R sentences [3:5] and [5:7], probe [7:9].
    mapping = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3])
    ones = torch.ones(4, 4)
    strongest_sweep = ones.clone()
    strongest_sweep[1:3, :3] = 0
    strongest_sweep[3, :1] = 0
    isolated = strongest_sweep.clone()
    isolated[3, :3] = 0
    native = forward(original_ids)
    handles = install_clean_sdpa_forward(model)
    try:
        clean = forward(original_ids)
        allones = masked_forward(original_ids, ones, mapping)
        masked = masked_forward(original_ids, strongest_sweep, mapping)
        iso = masked_forward(original_ids, isolated, mapping)
        alt_placeholder = original_ids.clone()
        alt_placeholder[0, -1] = 49
        placeholder_logits = forward(alt_placeholder)
        alt_prefix = original_ids.clone()
        alt_prefix[0, :7] = torch.arange(32, 39)
        alt_iso = masked_forward(alt_prefix, isolated, mapping)

        # A -1 token after P reads P and remains readable by every later token.
        relay_mapping = torch.tensor([0, 0, 0, -1, 1, 1, 1, 2, 2])
        relay_mask = torch.eye(3)
        changed_prompt = original_ids.clone()
        changed_prompt[0, :3] = torch.tensor([32, 33, 34])
        relay_a = masked_forward(original_ids, relay_mask, relay_mapping)
        relay_b = masked_forward(changed_prompt, relay_mask, relay_mapping)
        # Give the relay a real sentence index and block its P reads.
        controlled_mapping = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3, 3])
        controlled_mask = torch.eye(4)
        controlled_mask[2:, 1] = 1
        controlled_a = masked_forward(original_ids, controlled_mask, controlled_mapping)
        controlled_b = masked_forward(changed_prompt, controlled_mask, controlled_mapping)
    finally:
        remove_handles(handles)

    handles = install_mask_hooks(model, layers,
                                 {i: strongest_sweep[None].expand(heads, -1, -1) for i in layers},
                                 mapping, frozen_mask(4), True)
    try:
        eager_masked = forward(original_ids)
    finally:
        remove_handles(handles)
    expanded = expand_sentence_mask_to_tokens(strongest_sweep[None].expand(heads, -1, -1),
                                             mapping, frozen_mask(4), 9, 9)

    # Direct exact-zero gates have zero gradient through clamp_min in log masks.
    class MaskState:
        pass
    state = MaskState()
    state._circuit_mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]], requires_grad=True)
    state._token_to_sent = torch.tensor([0, 1])
    state._gap_filter = frozen_mask(2)
    _expand_mask_to_log_additive(state, 2, 2, None, torch.float32).sum().backward()
    exact_zero_gradient = float(state._circuit_mask.grad[0, 1, 0])

    metrics = {
        'native_eager_vs_clean_sdpa_maxabs': max_diff(native, clean),
        'clean_vs_allones_maxabs': max_diff(clean, allones),
        'masked_eager_vs_masked_sdpa_maxabs': max_diff(eager_masked, masked),
        'mask_effect_full_logits_maxabs': max_diff(clean, masked),
        'mask_effect_answer_logits_maxabs': max_diff(clean[0, -2], masked[0, -2]),
        'placeholder_swap_effect_preceding_logits_maxabs': max_diff(clean[:, :-1], placeholder_logits[:, :-1]),
        'isolated_probe_prefix_change_effect_maxabs': max_diff(iso[:, 7:], alt_iso[:, 7:]),
        'expanded_reasoning_to_prompt_max': float(expanded[:, 3:7, :3].max()),
        'expanded_probe_to_reasoning_min': float(expanded[:, 7:, 3:7].min()),
        'expanded_reasoning_within_sentence_min': float(expanded[:, 3:5, 3:5].min()),
        'unassigned_relay_prompt_change_effect_probe_maxabs': max_diff(relay_a[:, 7:], relay_b[:, 7:]),
        'relay_incoming_prompt_blocked_change_effect_probe_maxabs': max_diff(controlled_a[:, 7:], controlled_b[:, 7:]),
        'direct_zero_gate_log_mask_gradient': exact_zero_gradient,
    }
    checks = {
        'native_sdpa_parity': metrics['native_eager_vs_clean_sdpa_maxabs'] < 1e-5,
        'allones_identity': metrics['clean_vs_allones_maxabs'] == 0,
        'masked_backend_parity': metrics['masked_eager_vs_masked_sdpa_maxabs'] < 1e-5,
        'mask_has_effect': metrics['mask_effect_full_logits_maxabs'] > 1e-4,
        'causal_placeholder': metrics['placeholder_swap_effect_preceding_logits_maxabs'] == 0,
        'fully_isolated_probe_invariant': metrics['isolated_probe_prefix_change_effect_maxabs'] == 0,
        'requested_region_zero': metrics['expanded_reasoning_to_prompt_max'] == 0,
        'sweep_leaves_probe_to_reasoning': metrics['expanded_probe_to_reasoning_min'] == 1,
        'sweep_leaves_within_sentence_reads': metrics['expanded_reasoning_within_sentence_min'] == 1,
        'unassigned_relay_carries_information': metrics['unassigned_relay_prompt_change_effect_probe_maxabs'] > 1e-4,
        'blocked_relay_invariant': metrics['relay_incoming_prompt_blocked_change_effect_probe_maxabs'] == 0,
        'direct_zero_gate_no_gradient': exact_zero_gradient == 0,
    }
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Random tiny Qwen3 on CPU float32; does not validate trained Qwen3-8B GPU numerics or behavior.',
        'seed': args.seed,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__},
        'model_config': config.to_dict(), 'device': 'cpu', 'dtype': 'float32',
        'reproduce': f'OMP_NUM_THREADS=2 .venv/bin/python -m expts.prompt_bias_circuit_discovery.audit_mask_hooks --seed {args.seed} --output {args.output}',
        'metrics': metrics, 'checks': checks, 'passed': all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'metrics': metrics, 'checks': checks, 'passed': result['passed']}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
