"""Cache hidden-state activations of stored rollouts on disk, once per
(model, trace, position set), with an append-only index so that several
runs can cache independently without overwriting each other.

For every record (``prompt_token_ids`` + ``output_token_ids``) one forward
pass over prompt + output stores, for every layer (embeddings and the L
block outputs) and every requested position, the hidden state in float16:

- ``last_reasoning``: the last token before ``</think>``;
- ``think``: the ``</think>`` token itself;
- ``mean_reasoning``: the mean over the reasoning tokens (``<think>`` up to
  but excluding ``</think>``);
- ``last_output``: the last token of the whole output (the final answer).

Storage: ``<out_dir>/<run_id>_task<k>.npy`` with shape
(n_traces, n_positions, n_layers, hidden) and a sidecar ``.json`` listing
the rows; ``<out_dir>/index.jsonl`` maps the cache key (sha1 of model name,
prompt ids, output ids and position spec) to file and row. ``load_cached``
reads back any subset of keys through the index.

Usage:
    uv run python -m expts.prompt_bias_circuit_discovery.cache_activations \
        --records results/prompt_bias_v2/intervention_eval/qwen3_8b_admission_black_vs_white_confirmed_k32.json \
        --out_dir results/activations/qwen3_8b --run_id admission_k32 --task 0 --n_tasks 4
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from expts.direct_answer_circuit_discovery.learn import load_model_eager

POSITIONS = ["last_reasoning", "think", "mean_reasoning", "last_output"]


def cache_key(model_name, prompt_ids, output_ids, positions):
    h = hashlib.sha1()
    h.update(model_name.encode()); h.update(b"|"); h.update(json.dumps(list(prompt_ids)).encode()); h.update(b"|")
    h.update(json.dumps(list(output_ids)).encode()); h.update(b"|"); h.update(",".join(positions).encode())
    return h.hexdigest()


def read_index(out_dir):
    path = os.path.join(out_dir, "index.jsonl")
    idx = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line); idx[r["key"]] = r
    return idx


def append_index(out_dir, rows):
    path = os.path.join(out_dir, "index.jsonl")
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def load_cached(out_dir, keys, positions=None, layers=None):
    """Return an array (len(keys), n_pos, n_layers, hidden) for cached keys (missing keys raise)."""
    idx = read_index(out_dir)
    by_file = {}
    for k in keys:
        r = idx[k]; by_file.setdefault(r["file"], []).append((k, r["row"]))
    out = {}
    for f, items in by_file.items():
        arr = np.load(os.path.join(out_dir, f), mmap_mode="r")
        meta = json.load(open(os.path.join(out_dir, f[:-4] + ".json")))
        p_sel = [meta["positions"].index(p) for p in (positions or meta["positions"])]
        l_sel = list(range(arr.shape[2])) if layers is None else layers
        for k, row in items:
            out[k] = np.asarray(arr[row][p_sel][:, l_sel])
    return np.stack([out[k] for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, nargs="+")
    ap.add_argument("--model_name", default="Qwen/Qwen3-8B")
    ap.add_argument("--positions", nargs="+", default=POSITIONS)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=1)
    ap.add_argument("--flush_every", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs = []
    for p in args.records:
        recs.extend(json.load(open(p)))
    recs = recs[args.task::args.n_tasks]
    idx = read_index(args.out_dir)
    todo = []
    for r in recs:
        k = cache_key(args.model_name, r["prompt_token_ids"], r["output_token_ids"], args.positions)
        if k not in idx:
            todo.append((k, r))
    print(f"task {args.task}/{args.n_tasks}: {len(recs)} records, {len(todo)} not yet cached", flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(args.model_name)
    think_id = tok.convert_tokens_to_ids("</think>")
    model, _ = load_model_eager(args.model_name, device="cuda")
    dev = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    buf, rows, part = [], [], 0
    t0 = time.time()

    def flush():
        nonlocal buf, rows, part
        if not buf:
            return
        fname = f"{args.run_id}_task{args.task}_part{part}.npy"
        np.save(os.path.join(args.out_dir, fname), np.stack(buf).astype(np.float16))
        json.dump({"positions": args.positions, "model_name": args.model_name, "rows": rows}, open(os.path.join(args.out_dir, fname[:-4] + ".json"), "w"))
        append_index(args.out_dir, [dict(key=r["key"], tag=r["tag"], file=fname, row=i, positions=args.positions, model_name=args.model_name, n_layers=n_layers, hidden=hidden)
                                    for i, r in enumerate(rows)])
        print(f"  wrote {fname} ({len(rows)} traces, {time.time() - t0:.0f}s)", flush=True)
        buf, rows, part = [], [], part + 1

    for k, r in todo:
        p_ids, o_ids = list(r["prompt_token_ids"]), list(r["output_token_ids"])
        full = torch.tensor([p_ids + o_ids], device=dev)
        pl = len(p_ids)
        tpos = pl + (o_ids.index(think_id) if think_id in o_ids else len(o_ids) - 1)
        with torch.no_grad():
            hs = model(full, output_hidden_states=True).hidden_states  # (L+1) x (1, T, H)
        H = torch.stack([h[0] for h in hs], dim=0)  # (L+1, T, H)
        feats = []
        for pos in args.positions:
            if pos == "last_reasoning":
                v = H[:, max(pl, tpos - 1)]
            elif pos == "think":
                v = H[:, tpos]
            elif pos == "mean_reasoning":
                v = H[:, pl:max(pl + 1, tpos)].float().mean(dim=1)
            elif pos == "last_output":
                v = H[:, full.shape[1] - 1]
            else:
                raise ValueError(pos)
            feats.append(v.float().cpu().numpy())
        buf.append(np.stack(feats))  # (n_pos, L+1, H)
        rows.append(dict(key=k, tag=r.get("tag"), uid=r.get("uid"), n_tokens=int(full.shape[1]), think_pos=int(tpos)))
        del hs, H, full
        if len(buf) >= args.flush_every:
            flush()
    flush()
    print(f"task {args.task}: done {len(todo)} traces in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
