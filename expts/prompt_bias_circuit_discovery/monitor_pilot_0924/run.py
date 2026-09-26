"""Bounded, resumable Qwen3-32B pilot profiles. No judge calls or OOD selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM
from .prepare import ROOT, MODEL, REVISION, CONDITIONS, digest
from .core import CellPool, AttentionMask, trace_nll, sample_gates, size_loss, size_coefficient


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sync():
    torch.cuda.synchronize()


def checkpoint_save(path, alpha, optimizer, step, record, condition):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(alpha=alpha.detach().cpu(), optimizer=optimizer.state_dict(), step=step,
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    frozen_sha256=record['frozen_sha256'], condition=condition), tmp)
    tmp.replace(path)


@torch.no_grad()
def diagnostics(model, controller, record, pool, gates):
    ids = torch.tensor([record['input_ids']], device='cuda')
    controller.bias = pool.additive(gates) if pool is not None else None
    loss = float(trace_nll(model, ids, record['prompt_length']))
    full = torch.cat([ids, ids.new_tensor([[151668, 271, 334, 19357, 4226, 320]])], dim=1)
    if pool is not None:
        probe_pool = CellPool(record, pool.condition, 'cuda', with_probe=True)
        assert probe_pool.pairs == pool.pairs
        controller.bias = probe_pool.additive(gates)
    h = model.model(full, use_cache=False).last_hidden_state[:, -1:]
    logits = model.lm_head(h)[0, 0].float()
    raw = logits.softmax(-1)[[32, 33]].tolist()
    cond = logits[[32, 33]].softmax(-1).tolist()
    return dict(trace_nll=loss, option_mass=sum(raw), raw_ab=raw, conditional_ab=cond)


def fit(model, controller, record, condition, steps, directory):
    pool = CellPool(record, condition, 'cuda')
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    alpha = torch.full((pool.count,), 2., device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=.1, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    checkpoint_path = directory / f'{condition}.pt'
    log_path = directory / f'{condition}_training.json'
    history = []
    start = 0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        assert state['frozen_sha256'] == record['frozen_sha256'] and state['condition'] == condition
        with torch.no_grad():
            alpha.copy_(state['alpha'].to('cuda'))
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        start = state['step']
        history = json.loads(log_path.read_text())[:start] if log_path.exists() else []
    ids = torch.tensor([record['input_ids']], device='cuda')
    model.train()  # Enables nonreentrant layer checkpoints; Qwen dropout is zero.
    for step in range(start, steps):
        sync(); begun = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses, active = [], []
        for _ in range(4):
            gates = sample_gates(alpha)
            controller.bias = pool.additive(gates)
            loss = trace_nll(model, ids, record['prompt_length'])
            (loss / 4).backward()
            losses.append(float(loss.detach())); active.append(int((gates.detach() > 0).sum()))
        assert torch.isfinite(alpha.grad).all(), 'Nonfinite task gradient'
        grad_norm = float(alpha.grad.norm())
        if step == 0:
            assert grad_norm > 0, 'Trace loss has no gradient to the eligible cells'
        size_grad, = torch.autograd.grad(size_coefficient(step) * size_loss(alpha), alpha)
        optimizer.step()
        with torch.no_grad():
            alpha.sub_(.1 * size_grad)
        assert torch.isfinite(alpha).all(), 'Nonfinite gate parameters'
        controller.bias = None
        sync()
        row = dict(step=step + 1, nll=sum(losses)/4, sampled_active=sum(active)/4,
                   expected_active=float(torch.sigmoid(alpha.detach() - (2/3)*__import__('math').log(.1/1.1)).sum()),
                   task_gradient_norm=grad_norm, size_coefficient=size_coefficient(step),
                   seconds=time.monotonic()-begun, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        history.append(row)
        if (step + 1) % 5 == 0 or step + 1 == steps:
            write_json(log_path, history)
            checkpoint_save(checkpoint_path, alpha, optimizer, step + 1, record, condition)
            print(record['sample_id'], condition, json.dumps(row), flush=True)
    model.eval()
    binary = pool.hard(alpha)
    result = dict(steps=steps, eligible_cells=pool.count, kept_cells=int(binary.sum()),
                  structural_zero_task_gradient_cells=pool.structural_zero_cells(),
                  pairs=[[record['chunks'][q]['id'], record['chunks'][k]['id']] for q, k in pool.pairs],
                  binary=binary.cpu().tolist(), alpha=alpha.detach().cpu().tolist(),
                  diagnostics=diagnostics(model, controller, record, pool, binary),
                  total_training_seconds=sum(x['seconds'] for x in history),
                  mean_step_seconds=sum(x['seconds'] for x in history)/len(history),
                  final=(steps == 1000))
    write_json(directory / f'{condition}_mask.json', result)
    return result


@torch.no_grad()
def thought_anchors(model, controller, record, background, directory, source_limit):
    pool = CellPool(record, background, 'cuda')
    path = directory / f'ta_{background}.json'
    result = json.loads(path.read_text()) if path.exists() else dict(
        background=background, frozen_sha256=record['frozen_sha256'], columns={}, seconds_by_source={})
    assert result['frozen_sha256'] == record['frozen_sha256']
    ids = torch.tensor([record['input_ids']], device='cuda')
    base = pool.additive(torch.ones(pool.count, device='cuda'))
    controller.bias = base
    reference = model.model(ids, use_cache=False).last_hidden_state
    # Source choice is deterministic and only affects runtime profiling, never graph selection.
    sources = sorted({k for q, k in pool.pairs})
    chosen = sources if source_limit == 0 else sources[:source_limit]
    t = torch.arange(len(record['input_ids']), device='cuda')
    for source in chosen:
        if str(source) in result['columns']:
            continue
        sync(); begun = time.monotonic()
        chunk = record['chunks'][source]
        block = (t[None, :] >= max(3, chunk['start'])) & (t[None, :] <= chunk['end']) & (t[None, :] < t[:, None])
        controller.bias = base.masked_fill(block[None, None], float('-inf'))
        changed = model.model(ids, use_cache=False).last_hidden_state
        kl = []
        for start in range(0, ids.shape[1], 64):
            a = model.lm_head(reference[:, start:start+64]).float().log_softmax(-1)
            b = model.lm_head(changed[:, start:start+64]).float().log_softmax(-1)
            kl.append((a.exp() * (a-b)).sum(-1).squeeze(0))
        values = torch.cat(kl)
        result['columns'][str(source)] = [float(values[c['start']:c['end']+1].mean()) for c in record['chunks']]
        sync()
        result['seconds_by_source'][str(source)] = time.monotonic() - begun
        result['total_sources'] = len(sources)
        result['complete'] = len(result['columns']) == len(sources)
        write_json(path, result)
        print(record['sample_id'], 'TA', background, source, result['seconds_by_source'][str(source)], flush=True)
    controller.bias = None
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', required=True)
    ap.add_argument('--steps', type=int, default=50, choices=range(1, 1001))
    ap.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    ap.add_argument('--ta-sources', type=int, default=2, help='0 means complete TA; positive values are runtime-only profiles')
    args = ap.parse_args()
    start = time.monotonic()
    directory = ROOT / args.sample
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'frozen_inputs.json').read_text())
    record = next(r for r in manifest['records'] if r['sample_id'] == args.sample)
    check = dict(record); sha = check.pop('frozen_sha256')
    assert digest(check) == sha
    assert manifest['model'] == MODEL and manifest['revision'] == REVISION
    code_hash = hashlib.sha256(b''.join((Path(__file__).parent / f).read_bytes() for f in ['core.py', 'run.py'])).hexdigest()
    metadata_path = directory / 'metadata.json'
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        assert previous['implementation_sha256'] == code_hash, 'Implementation changed; review before resuming'
    metadata = dict(sample_id=args.sample, frozen_sha256=sha, model=MODEL, revision=REVISION,
                    implementation_sha256=code_hash, torch=torch.__version__, transformers=transformers.__version__,
                    python=platform.python_version(), gpu=torch.cuda.get_device_name(),
                    attention='explicit eager; FP32 softmax; exact -inf zero gates',
                    job_id=os.environ.get('SLURM_JOB_ID'), requested_steps=args.steps,
                    conditions=args.conditions, ta_profile_source_limit=args.ta_sources)
    write_json(metadata_path, metadata)
    print(json.dumps(metadata), flush=True)
    assert 'H100' in metadata['gpu'], metadata['gpu']
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                  torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager')
    model.config.use_cache = False
    assert model.config.attention_dropout == 0
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    ids = torch.tensor([record['input_ids']], device='cuda')
    with torch.no_grad():
        native_nll = float(trace_nll(model, ids, record['prompt_length']))
        native_hidden = model.model(ids, use_cache=False).last_hidden_state
    controller = AttentionMask(model)
    with torch.no_grad():
        controller.bias = CellPool(record, 'joint', 'cuda').additive(torch.ones(CellPool(record, 'joint').count, device='cuda'))
        actual_hidden = model.model(ids, use_cache=False).last_hidden_state
        delta = float((actual_hidden.float()-native_hidden.float()).abs().max())
        assert torch.equal(actual_hidden, native_hidden), f'All-one mask disagrees with native eager: max delta {delta}'
    del actual_hidden, native_hidden
    controller.bias = None
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    metadata['native_nll'] = native_nll
    metadata['all_one_hidden_max_abs_error'] = delta
    metadata['load_and_identity_seconds'] = time.monotonic()-start
    write_json(metadata_path, metadata)
    control_path = directory / 'controls.json'
    controls = json.loads(control_path.read_text()) if control_path.exists() else {}
    if 'clean' not in controls:
        controls['clean'] = diagnostics(model, controller, record, None, None)
        assert abs(controls['clean']['trace_nll']-native_nll) < 1e-6
    for condition in args.conditions:
        pool = CellPool(record, condition, 'cuda')
        for label, gates in [('full', torch.ones(pool.count, device='cuda')),
                             ('empty', torch.zeros(pool.count, device='cuda'))] + [
                             (f'random{seed}', pool.random(seed)) for seed in range(5)]:
            key = condition + '/' + label
            if key not in controls:
                controls[key] = diagnostics(model, controller, record, pool, gates)
                write_json(control_path, controls)
                print(args.sample, key, controls[key], flush=True)
        fit(model, controller, record, condition, args.steps, directory)
    backgrounds = ['joint', 'rr_off', 'pr_off'] if len(args.conditions) > 1 else ['joint']
    for background in backgrounds:
        thought_anchors(model, controller, record, background, directory, args.ta_sources)
    metadata['elapsed_seconds'] = time.monotonic() - start
    metadata['allocated_gpu_hours_this_invocation'] = metadata['elapsed_seconds'] / 3600
    metadata['status'] = 'profile_complete' if args.steps < 1000 or args.ta_sources else 'fits_and_TA_complete'
    write_json(directory / f'completion_{os.environ.get("SLURM_JOB_ID", "local")}.json', metadata)
    print('COMPLETE', json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
