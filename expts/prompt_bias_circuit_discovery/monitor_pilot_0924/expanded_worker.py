"""Independent expanded-study task: one trace/condition or one TA background per GPU."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from .prepare import MODEL, REVISION, digest
from .core import AttentionMask, CellPool, trace_nll
from .run import fit, thought_anchors, diagnostics, write_json

EXPANDED = Path('results/prompt_bias_v2/monitor_expanded_0924')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--index', type=int)
    args = ap.parse_args()
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    task = json.loads(Path(args.manifest).read_text())[index]
    record = json.loads((EXPANDED/'inputs'/f"{task['sample_id']}.json").read_text())
    check = dict(record); sha = check.pop('frozen_sha256')
    assert digest(check) == sha
    directory = EXPANDED / task['kind'] / task['sample_id'] / task['condition']
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory/'complete.json').exists(), 'Refusing duplicate completed work'
    if (EXPANDED/'STOP').exists():
        raise RuntimeError('Expanded sweep STOP sentinel exists')
    begun = time.monotonic()
    meta = dict(task=task, frozen_sha256=sha, model=MODEL, revision=REVISION,
                job_id=os.environ.get('SLURM_JOB_ID'), array_task=index,
                gpu=torch.cuda.get_device_name(), status='running')
    meta['implementation_sha256'] = hashlib.sha256(b''.join((Path(__file__).parent/f).read_bytes()
                      for f in ['core.py','run.py','expanded_worker.py'])).hexdigest()
    write_json(directory/'metadata.json', meta)
    try:
        assert 'H100' in meta['gpu']
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                    torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager')
        model.config.use_cache = False
        assert model.config.attention_dropout == 0
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        ids = torch.tensor([record['input_ids']], device='cuda')
        with torch.no_grad():
            reference = model.model(ids, use_cache=False).last_hidden_state
        controller = AttentionMask(model)
        with torch.no_grad():
            pool = CellPool(record, 'joint', 'cuda')
            controller.bias = pool.additive(torch.ones(pool.count, device='cuda'))
            actual = model.model(ids, use_cache=False).last_hidden_state
            assert torch.equal(reference, actual), 'Native attention identity failed'
        del reference, actual
        controller.bias = None
        meta['all_one_hidden_max_abs_error'] = 0.
        if task['kind'] == 'fits':
            pool = CellPool(record, task['condition'], 'cuda')
            controls = {'clean':diagnostics(model, controller, record, None, None)}
            for label, mask in [('full',torch.ones(pool.count,device='cuda')),
                                ('empty',torch.zeros(pool.count,device='cuda'))] + [
                                (f'random{i}',pool.random(i)) for i in range(5)]:
                controls[label] = diagnostics(model,controller,record,pool,mask)
            write_json(directory/'controls.json',controls)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.enable_input_require_grads()
            # Check the user-authorized stop switch at every optimizer update.
            original_step = torch.optim.Adam.step
            def guarded_step(optimizer, *step_args, **step_kwargs):
                if (EXPANDED/'STOP').exists():
                    raise RuntimeError('Expanded sweep stopped after a recorded failure')
                return original_step(optimizer, *step_args, **step_kwargs)
            torch.optim.Adam.step = guarded_step
            fit(model,controller,record,task['condition'],1000,directory)
        else:
            assert task['kind'] == 'ta'
            thought_anchors(model,controller,record,task['condition'],directory,0)
        meta.update(status='complete',elapsed_seconds=time.monotonic()-begun)
        write_json(directory/'complete.json',meta)
    except Exception as exc:
        meta.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                    elapsed_seconds=time.monotonic()-begun)
        write_json(directory/'failure.json',meta)
        # Integrity/numerical failures invalidate other work; OOM is local to that length.
        catastrophic = isinstance(exc, AssertionError) or 'Nonfinite' in str(exc)
        if catastrophic:
            (EXPANDED/'STOP').write_text(json.dumps(meta,indent=2))
        raise


if __name__ == '__main__':
    main()
