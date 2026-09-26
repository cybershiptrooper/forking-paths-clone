"""One independent GPU evaluation per fixed trace/condition, safe for Slurm arrays."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from .prepare import ROOT, MODEL, REVISION, CONDITIONS, digest
from .core import CellPool, AttentionMask
from .run import diagnostics, write_json


def select_ta(record, condition, directory):
    pool = CellPool(record, condition)
    background = condition if condition.endswith('_off') else 'joint'
    ta = json.loads((directory / f'ta_{background}.json').read_text())
    assert ta['complete'] and ta['frozen_sha256'] == record['frozen_sha256']
    scores = torch.tensor([ta['columns'][str(source)][reader] for reader, source in pool.pairs])
    assert torch.isfinite(scores).all()
    return pool.hard(scores), scores, background


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', required=True)
    ap.add_argument('--condition', choices=CONDITIONS, required=True)
    args = ap.parse_args()
    begun = time.monotonic()
    directory = ROOT / args.sample
    output = directory / f'{args.condition}_ta_mask.json'
    # Independent tasks write disjoint files. Never run a duplicate evaluation silently.
    assert not output.exists(), output
    record = next(r for r in json.loads((ROOT/'frozen_inputs.json').read_text())['records'] if r['sample_id'] == args.sample)
    checked = dict(record); sha = checked.pop('frozen_sha256')
    assert digest(checked) == sha
    snp = json.loads((directory / f'{args.condition}_mask.json').read_text())
    assert snp['final'] and snp['steps'] == 1000
    binary, scores, background = select_ta(record, args.condition, directory)
    pool = CellPool(record, args.condition, 'cuda')
    pairs = [[record['chunks'][q]['id'], record['chunks'][k]['id']] for q, k in pool.pairs]
    assert pairs == snp['pairs'] and int(binary.sum()) == snp['kept_cells']
    gpu = torch.cuda.get_device_name()
    assert 'H100' in gpu
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                    torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager')
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller = AttentionMask(model)
    result = dict(sample_id=args.sample, condition=args.condition, background=background,
                  frozen_sha256=sha, method='thought_anchors', eligible_cells=pool.count,
                  kept_cells=int(binary.sum()), pairs=pairs, binary=binary.tolist(), scores=scores.tolist(),
                  diagnostics=diagnostics(model, controller, record, pool, binary.to('cuda')),
                  model=MODEL, revision=REVISION, gpu=gpu, job_id=os.environ.get('SLURM_JOB_ID'),
                  array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), final=True)
    result['elapsed_seconds'] = time.monotonic()-begun
    write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','binary','scores']}), flush=True)


if __name__ == '__main__':
    main()
