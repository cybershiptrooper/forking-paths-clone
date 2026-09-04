"""Attention-based baselines from the external task.

Two selection baselines, both reading attention weights at one layer
(their next-sentence attention reads layer 42 of Qwen3-32B's 64 layers):

- ``attn_last``  — score each compress-region sentence by the attention it
  receives from the last sentence of the prefix (their "Attention (last
  prefix sentence)").
- ``attn_next``  — score by the attention it receives from the forced next
  sentence, appended after the prefix (their "next-sentence attention").

Implementation: the model runs with SDPA everywhere (no attention weights
materialised); a forward pre-hook on the scoring layer captures its input
hidden states and position embeddings, and Q/K are recomputed manually for
just the needed query rows.  Score[j] = mean over heads and query rows of
the summed attention mass onto sentence j's key tokens.

Usage (GPU):
    uv run python -m expts.external_compression.baselines \
        --instance gpqa_gpqa_diamond_0001_pl60
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

import torch

from expts.external_compression.common import (
    DATA_DIR,
    K_KEEP,
    MODEL_NAME,
    RESULTS_DIR,
    all_sentence_token_ranges,
    load_rollout,
)

SCORES_DIR = os.path.join(RESULTS_DIR, "scores")
# Their choice: layer 42 of Qwen3-32B's 64 layers.  Env override exists for
# smoke tests on smaller models.
ATTN_LAYER = int(os.environ.get("EXTCOMP_ATTN_LAYER", "42"))


def captured_attention_rows(
    model, input_ids: torch.Tensor, layer_idx: int, row_range: Tuple[int, int],
) -> torch.Tensor:
    """Attention weights (heads, num_rows, k_len) at *layer_idx* for the
    query rows in the inclusive *row_range*, recomputed from captured
    hidden states.  Causal within the sequence."""
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    captured = {}

    def pre_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        pe = kwargs.get("position_embeddings")
        if pe is None:
            for a in args:
                if isinstance(a, tuple) and len(a) == 2:
                    pe = a
                    break
        captured["hidden_states"] = hs.detach()
        captured["position_embeddings"] = tuple(t.detach() for t in pe)

    handle = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    with torch.no_grad():
        model(input_ids)
    handle.remove()

    hs = captured["hidden_states"]           # (1, T, D)
    cos, sin = captured["position_embeddings"]
    bsz, T, _ = hs.shape
    head_dim = attn.head_dim
    hidden_shape = (bsz, T, -1, head_dim)

    with torch.no_grad():
        q = attn.q_norm(attn.q_proj(hs).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hs).view(hidden_shape)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        n_rep = q.shape[1] // k.shape[1]
        k = repeat_kv(k, n_rep)              # (1, H, T, hd)

        r0, r1 = row_range
        q_rows = q[:, :, r0 : r1 + 1, :]     # (1, H, R, hd)
        scores = torch.matmul(q_rows, k.transpose(-1, -2)) * attn.scaling
        # Causal mask: query at absolute position p attends to keys <= p.
        R = r1 - r0 + 1
        key_pos = torch.arange(T, device=scores.device).view(1, 1, 1, T)
        query_pos = torch.arange(r0, r1 + 1, device=scores.device).view(1, 1, R, 1)
        scores = scores.masked_fill(key_pos > query_pos, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1)  # (1, H, R, T)
    return weights[0]


def attention_scores(
    model, ids: List[int], ranges: List[Tuple[int, int]],
    query_range: Tuple[int, int], num_compress: int,
) -> List[float]:
    """Per-compress-sentence attention received from *query_range* rows."""
    device = next(model.parameters()).device
    input_t = torch.tensor([ids], device=device)
    weights = captured_attention_rows(model, input_t, ATTN_LAYER, query_range)
    # ranges[1 .. num_compress] are the compress-region sentences.
    scores = [0.0]  # index 0 = prompt block, not rankable
    for j in range(1, num_compress + 1):
        a, b = ranges[j]
        mass = weights[:, :, a : b + 1].sum(dim=-1)   # (H, R)
        scores.append(float(mass.mean().item()))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--model", dest="model_obj", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run(args.instance, model_name=args.model_name)


def run(instance_id: str, model=None, tokenizer=None, model_name: str = MODEL_NAME):
    with open(os.path.join(DATA_DIR, "instances.json")) as f:
        instances = {r["instance_id"]: r for r in json.load(f)}
    inst = instances[instance_id]
    with open(os.path.join(DATA_DIR, "prompts_rendered.json")) as f:
        rendered = json.load(f)

    qid = inst["question_id"]
    N = inst["prefix_length"]
    roll = load_rollout(qid)
    sents = roll["sentences"][:N]
    next_sent = roll["sentences"][N]
    prompt_str = rendered[qid]["prompt_str"]
    num_compress = N - K_KEEP

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa",
        )
        model.eval()

    out_dir = os.path.join(SCORES_DIR, instance_id)
    os.makedirs(out_dir, exist_ok=True)

    # attn_last: input = prefix only; query rows = last prefix sentence.
    t0 = time.time()
    ids, ranges = all_sentence_token_ranges(tokenizer, prompt_str, sents)
    scores_last = attention_scores(model, ids, ranges, ranges[N], num_compress)
    _write(out_dir, instance_id, "attn_last", scores_last, num_compress, t0)

    # attn_next: input = prefix + forced next sentence; query rows = next sent.
    t0 = time.time()
    ids2, ranges2 = all_sentence_token_ranges(
        tokenizer, prompt_str, sents, extra_text=" " + next_sent,
    )
    scores_next = attention_scores(model, ids2, ranges2, ranges2[-1], num_compress)
    _write(out_dir, instance_id, "attn_next", scores_next, num_compress, t0)


def _write(out_dir, instance_id, method, scores, num_rankable, t0):
    rec = {
        "instance_id": instance_id,
        "method": method,
        "scores": scores,
        "num_rankable": num_rankable,
        "attn_layer": ATTN_LAYER,
        "seconds": time.time() - t0,
    }
    path = os.path.join(out_dir, f"{method}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"wrote {path} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
