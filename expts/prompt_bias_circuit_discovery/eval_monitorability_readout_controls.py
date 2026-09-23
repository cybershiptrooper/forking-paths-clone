"""Small fixed-trace diagnostics separating prefix masks from answer readout.

This is teacher-forced replay, not a causal/on-policy generation experiment.
Every attention block is applied at every layer/head; token self-attention is
preserved. Direct token masks bypass production sentence-to-token expansion,
while reusing its architecture-specific Q/K/V and SDPA implementation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import types

import torch
from transformers import AutoTokenizer

from utils.circuit_eval import install_clean_sdpa_forward, remove_handles
from utils.circuit_discovery.sdpa_forward import make_sdpa_attention_forward
from utils.masks import build_gap_filter, build_mode_filter, build_causal_filter, build_combined_filter
from utils.utils import get_attention_module, set_seed
from expts.direct_answer_circuit_discovery.learn import _build_prefix, load_model_eager
from expts.direct_answer_circuit_discovery.eval_log_alpha import _evaluate_mask
from expts.direct_answer_circuit_discovery.probe import build_answer_probe
from expts.prompt_bias_circuit_discovery.gen_mask_configs import PROBE_SUFFIX


def metrics(model, full, probe, prefix_len, **kwargs):
    with torch.inference_mode():
        # Only the answer position is needed; avoid the full sequence x vocab logits.
        result = model(full, use_cache=False, logits_to_keep=full.shape[-1] - probe.answer_logit_position(prefix_len), **kwargs)
        row = result.logits[0, 0].float()
        answer_ids = probe.answer_token_ids.to(row.device)
        raw = row.softmax(-1)[answer_ids].cpu().tolist()
        cond = row[answer_ids].softmax(-1).cpu().tolist()
        top = int(row.argmax())
    return dict(raw=raw, answer_mass=sum(raw), conditional=cond, top_token_id=top)


def install_direct(model, blocked):
    """A Boolean token-by-token block list, independent of sentence expansion."""
    additive = torch.zeros_like(blocked, dtype=torch.bfloat16).masked_fill(blocked, torch.finfo(torch.bfloat16).min)[None, None]

    def converter(module, q_len, k_len, cache_position, dtype):
        assert q_len == k_len == blocked.shape[0], (q_len, k_len, blocked.shape)
        return additive.to(dtype=dtype)

    fn = make_sdpa_attention_forward(model.config.model_type, mask_converter=converter)
    old = []
    for layer in range(model.config.num_hidden_layers):
        attn = get_attention_module(model, layer)
        old.append((attn, attn.forward))
        attn.forward = types.MethodType(fn, attn)
    return old


def direct_metrics(model, full, probe, prefix_len, blocked):
    assert not blocked.diagonal().any(), "Token diagonal must remain available"
    old = install_direct(model, blocked)
    try:
        return metrics(model, full, probe, prefix_len)
    finally:
        for attn, forward in old:
            attn.forward = forward


def production_metrics(model, full, probe, prefix_len, binary, token_to_sent, combined):
    attrs = ('_circuit_mask', '_token_to_sent', '_gap_filter')
    for layer in range(model.config.num_hidden_layers):
        attn = get_attention_module(model, layer)
        attn._circuit_mask = binary.unsqueeze(0).expand(model.config.num_attention_heads, -1, -1)
        attn._token_to_sent = token_to_sent
        attn._gap_filter = combined
        attn._renormalize_masked_attn = True
    try:
        return metrics(model, full, probe, prefix_len)
    finally:
        for layer in range(model.config.num_hidden_layers):
            attn = get_attention_module(model, layer)
            for attr in attrs:
                delattr(attn, attr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='results/prompt_bias_v2/reduce_masks/reduce_masks_qwen3_8b/dataset.json')
    ap.add_argument('--ids', type=int, nargs='+', default=[0, 1, 4, 5, 12, 13, 14, 15])
    ap.add_argument('--model_name', default='Qwen/Qwen3-8B')
    ap.add_argument('--out', default='results/prompt_bias_v2/readout_controls_0923/results.json')
    args = ap.parse_args()
    set_seed(42)
    dataset = json.load(open(args.dataset))
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model, _ = load_model_eager(args.model_name, device='cuda')
    model.eval()
    model.config.use_cache = False
    dev = next(model.parameters()).device
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(dataset=args.dataset, model=args.model_name, ids=args.ids, suffix=PROBE_SUFFIX, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
               note='Teacher-forced replay; attention renormalized; all layers/heads; same-sentence reads retained except self-only control.', rows=[], reversed_options=[])
    handles = install_clean_sdpa_forward(model)
    seen_prompts = set()
    start = time.monotonic()
    for i in args.ids:
        rec = dataset[i]
        letters = [x.strip() for x in rec['all_letters']]
        probe = build_answer_probe(tok, suffix=PROBE_SUFFIX, answer_letters=letters)
        for cut in ['prompt', 'half', 'think']:
            prefix, sentences, _, _, _, n_p = _build_prefix(
                tokenizer=tok, prompt=None, data_path=args.dataset, prompt_index=i, base_answer_type='stored',
                analysis_timestep=(0 if cut == 'prompt' else rec['analysis_timestep']) if cut != 'half' else None,
                analysis_sentence_step=rec['analysis_sentence_step_half'] if cut == 'half' else None,
                min_sentence_length=10, sentence_chunk=1, sentences_after_prefix=0)
            prefix_len = prefix.shape[-1]
            prompt_len = len(rec['prompt_token_ids'])
            full = torch.cat([prefix.to(dev), probe.make_continuation(dev)], -1)
            total = full.shape[-1]
            S = len(sentences)
            mapping = torch.full((total,), -1, device=dev, dtype=torch.long)
            for si, sent in enumerate(sentences):
                mapping[sent.start:sent.end + 1] = si
            mapping[prefix_len:] = S
            combined = build_combined_filter(build_gap_filter(S + 1, 1, device=dev),
                build_mode_filter(S, S + 1, 'both', device=dev), build_causal_filter(S + 1, device=dev), None)
            ix = torch.arange(S + 1, device=dev)
            p2r = ((ix[:, None] >= n_p) & (ix[:, None] < S) & (ix[None, :] < n_p) & ~combined)
            r2r = ((ix[:, None] >= n_p) & (ix[:, None] < S) & (ix[None, :] >= n_p) & (ix[None, :] < S) & ~combined)
            q2p = ((ix[:, None] == S) & (ix[None, :] < n_p) & ~combined)
            existing_pools = {'no_both': p2r | r2r, 'no_both_probe': p2r | r2r | q2p}
            positions = torch.arange(total, device=dev)
            q = positions[:, None]
            k = positions[None, :]
            causal = k <= q
            query_probe = q >= prefix_len
            prefix_key = k < prefix_len
            valid = (mapping[:, None] >= 0) & (mapping[None, :] >= 0)
            safe = mapping.clamp_min(0)
            zero = torch.zeros(total, total, device=dev, dtype=torch.bool)
            row = dict(example_id=i, uid=rec['uid'], positive=rec['is_positive'], trace_answer=rec['mask_target_letter'],
                       letters=letters, cut=cut, prefix_tokens=prefix_len, prompt_tokens=prompt_len,
                       n_reasoning_sentences=S - n_p,
                       unassigned_prefix=[dict(position=j, token=tok.decode([int(full[0,j])])) for j in torch.where(mapping[:prefix_len] < 0)[0].tolist()], conditions={}, validation={})
            cond = row['conditions']
            cond['clean'] = metrics(model, full, probe, prefix_len)
            cond['direct_identity'] = direct_metrics(model, full, probe, prefix_len, zero)
            ones = torch.ones(S + 1, S + 1, device=dev)
            cond['production_identity'] = production_metrics(model, full, probe, prefix_len, ones, mapping, combined)
            explicit = {}
            for name, pool in existing_pools.items():
                binary = ones.clone(); binary[pool] = 0
                cond['production_' + name] = production_metrics(model, full, probe, prefix_len, binary, mapping, combined)
                block = pool[safe[:, None], safe[None, :]] & valid & causal
                explicit[name] = block
                cond['direct_' + name] = direct_metrics(model, full, probe, prefix_len, block)
                row['validation'][name + '_conditional_max_error'] = max(abs(a-b) for a,b in zip(cond['production_' + name]['conditional'],cond['direct_' + name]['conditional']))
            # Check the exact public evaluation function on every cut, including
            # its full-logit path rather than only the optimized readout path.
            p = _evaluate_mask(model, list(range(model.config.num_hidden_layers)), model.config.num_attention_heads,
                              full, prefix_len, probe, ones, mapping, combined, dev, True, 'sdpa').tolist()
            row['validation']['evaluate_mask_identity_max_error'] = max(abs(a-b) for a,b in zip(p,cond['clean']['conditional']))
            # Every suffix query loses every prefix key. No -1/sink bypass.
            isolate = query_probe & prefix_key
            cond['probe_isolated'] = direct_metrics(model, full, probe, prefix_len, isolate)
            cond['probe_no_prompt'] = direct_metrics(model, full, probe, prefix_len, isolate & (k < prompt_len))
            cond['probe_no_reasoning'] = direct_metrics(model, full, probe, prefix_len, isolate & (k >= prompt_len))
            cond['no_both_probe_isolated'] = direct_metrics(model, full, probe, prefix_len, explicit['no_both_probe'] | isolate)
            cond['probe_isolated_keep_start3'] = direct_metrics(model, full, probe, prefix_len, isolate & (k >= 3))
            cond['probe_isolated_keep_unassigned'] = direct_metrics(model, full, probe, prefix_len, isolate & (mapping[None, :] >= 0))
            # Close the prompt->unassigned <think> relay while retaining only
            # the leading context-free 3-token chat sink. Within-sentence
            # reasoning reads remain; all unassigned tail tokens are covered.
            reason_query = (q >= prompt_len) & (q < prefix_len)
            prompt_key = k < prompt_len
            same_sentence = (mapping[:, None] == mapping[None, :]) & valid
            prior_reason = (k >= prompt_len) & (k < prefix_len) & (k < q) & ~same_sentence
            strict = ((reason_query & (prompt_key | prior_reason)) | (query_probe & prompt_key)) & (k >= 3) & causal
            strict.fill_diagonal_(False)
            cond['strict_no_both_probe_keep_start3'] = direct_metrics(model, full, probe, prefix_len, strict)
            # Extreme sanity control: each token only reads itself.
            cond['all_token_self_only'] = direct_metrics(model, full, probe, prefix_len, (k < q).expand(total, total))
            suffix = probe.make_continuation(dev)
            suffix_positions = torch.arange(prefix_len, prefix_len + suffix.shape[-1], device=dev)[None]
            cond['suffix_only_matching_rope'] = metrics(model, suffix, probe, 0, position_ids=suffix_positions)
            row['validation']['isolated_suffix_conditional_max_error'] = max(abs(a-b) for a,b in zip(cond['probe_isolated']['conditional'], cond['suffix_only_matching_rope']['conditional']))
            if i == args.ids[0] and cut == 'think':
                swapped = full.clone()
                swapped[0, -1] = probe.answer_token_ids[1]
                cond['placeholder_swapped'] = metrics(model, swapped, probe, prefix_len)
                row['validation']['placeholder_causality_max_error'] = max(abs(a-b) for a,b in zip(cond['clean']['conditional'], cond['placeholder_swapped']['conditional']))
            if i == args.ids[0] and cut == 'prompt':
                # Cross-check architecture replacement against native eager.
                remove_handles(handles)
                cond['native_eager_clean'] = metrics(model, full, probe, prefix_len)
                handles = install_clean_sdpa_forward(model)
                row['validation']['native_eager_conditional_max_error'] = max(abs(a-b) for a,b in zip(cond['clean']['conditional'], cond['native_eager_clean']['conditional']))
            out['rows'].append(row)
            out['elapsed_seconds'] = time.monotonic() - start
            out_path.write_text(json.dumps(out, indent=2))
            print(f"ex{i:03d} {cut:6s} tokens={prefix_len} clean={cond['clean']['conditional']} both_probe={cond['production_no_both_probe']['conditional']} isolated={cond['probe_isolated']['conditional']} mass={cond['probe_isolated']['answer_mass']:.5f} validation={row['validation']}", flush=True)
            del full, prefix, mapping, combined, explicit, zero, strict
            torch.cuda.empty_cache()
        if rec['uid'] not in seen_prompts:
            seen_prompts.add(rec['uid'])
            assert 'A) Yes\nB) No' in rec['prompt']
            reversed_text = rec['prompt'].replace('A) Yes\nB) No', 'A) No\nB) Yes')
            reversed_prefix = torch.tensor(tok.encode(reversed_text, add_special_tokens=False), device=dev)[None]
            reversed_full = torch.cat([reversed_prefix, probe.make_continuation(dev)], -1)
            reverse = dict(uid=rec['uid'], example_id=i, note='Prompt-only option reversal, no replayed reasoning.',
                           letters=letters, raw=metrics(model, reversed_full, probe, reversed_prefix.shape[-1]))
            out['reversed_options'].append(reverse)
            out_path.write_text(json.dumps(out, indent=2))
    remove_handles(handles)
    print(f'Finished {len(out["rows"])} rows in {time.monotonic() - start:.1f}s: {out_path}', flush=True)
    from expts.prompt_bias_circuit_discovery.report_monitorability_readout_controls import write_report
    write_report(out_path)


if __name__ == '__main__':
    main()
