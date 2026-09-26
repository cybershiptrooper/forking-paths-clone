"""Freeze exact expanded records concurrently, then submit independent ready tasks.

Public questions only go to the permitted segmenter. Completed pilot inputs are reused.
No sample substitutions, class filtering, or metric-informed decisions occur here.
"""
import concurrent.futures as cf
import json
import os
import subprocess
from pathlib import Path
from transformers import AutoTokenizer
from .prepare import ROOT, PROTOCOL, MODEL, REVISION, CONDITIONS, SEGMENT_MAX_TOKENS, payload, digest, prepare_record
from .expanded_worker import EXPANDED
from .run import write_json
from expts.prompt_bias_circuit_discovery.openrouter_client import get_client, load_cache


def append_frozen(records):
    all_records = {r['sample_id']:r for r in json.loads((ROOT/'frozen_inputs.json').read_text())['records']}
    all_records.update({r['sample_id']:r for r in records})
    frozen = [dict(sample_id=r['sample_id'], sha256=r['frozen_sha256'], prompt_length=r['prompt_length'],
                    prompt_char_spans=r['prompt_char_spans'], chunks=r['chunks']) for r in all_records.values()]
    text = PROTOCOL.read_text().split('\n\n## Appendix C. Frozen pilot chunks')[0].split('\n\n## Appendix C. Frozen experiment chunks')[0]
    appendix = ('\n\n## Appendix C. Frozen experiment chunks\n\n'
        'These exact character/token spans were frozen before fitting. Token ends are inclusive and character ends are exclusive. '
        'The original five pilot records are reused unchanged. For sarcasm, the parent comment and reply are segmented separately with the same semantic-unit instruction. '
        'On 24 September the user authorized the full fixed-cohort sweep without waiting for the earlier trace-review gate, with permission to stop on severe failure. '
        'No sample, objective, or sparsity changes follow from that scheduling authorization. The initial 4,000-token segmenter limit caused truncated JSON. Unfinished records receive at most two attempts with a 16,000-token limit, using the same instruction and reconstruction checks. Valid original segmentations remain fixed.\n\n'
        '<!-- BEGIN FROZEN PILOT CHUNKS -->\n```json\n' + json.dumps(frozen,ensure_ascii=False,indent=2) +
        '\n```\n<!-- END FROZEN PILOT CHUNKS -->\n')
    tmp = PROTOCOL.with_suffix('.md.tmp')
    tmp.write_text(text+appendix);tmp.replace(PROTOCOL)


def main():
    data = payload()['records']
    assert len(data) == 120
    client = get_client()
    cache = load_cache(ROOT/'segmenter_cache.json')
    tok = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    existing = json.loads((ROOT/'frozen_inputs.json').read_text())['records']
    for r in existing:
        write_json(EXPANDED/'inputs'/f"{r['sample_id']}.json",r)
    ready = {p.stem for p in (EXPANDED/'inputs').glob('*.json')}
    todo = [r for r in data if r['sample_id'] not in ready]
    # Same base across hint directions is processed serially; neutral text segmentation is cached.
    groups = {}
    for r in todo:
        groups.setdefault((r['task'],r['base_id']),[]).append(r)
    ledger_path = EXPANDED/'preparation_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(status='preparing',cohort_size=120,reused_samples=[r['sample_id'] for r in existing],
                  prepared=[],failures=[],submissions=[],concurrency=8,
                  authorized_expansion=True, validation_gate_waived_by_user=True)
    assert all(s['status']=='submitted' for s in ledger['submissions']), 'Resolve uncommitted scheduler intent before resume'
    ledger['status']='preparing'
    ledger['segmentation_output_limit']=SEGMENT_MAX_TOKENS
    ledger['output_limit_repair']='Original responses were truncated before the closing JSON bracket; reuse every valid cached segmentation, retry only unfinished records with the unchanged instruction.'
    write_json(ledger_path,ledger)
    reused={r['sample_id'] for r in existing}
    submitted=reused | {sid for batch in ledger['submissions'] for sid in batch['samples']}
    prepared=[json.loads(p.read_text()) for p in (EXPANDED/'inputs').glob('*.json') if p.stem not in reused]
    pending=[r for r in prepared if r['sample_id'] not in submitted]

    def group_prepare(group):
        out=[]
        for r in group:
            item=prepare_record(r,tok,client,cache)
            item['segmentation_output_limit']=SEGMENT_MAX_TOKENS
            item['frozen_sha256']=digest(item)
            out.append(item)
        return out

    def submit_batch():
        nonlocal pending
        if not pending:
            return
        append_frozen(prepared)
        tasks=[]
        for r in pending:
            tasks += [dict(sample_id=r['sample_id'],kind='fits',condition=c) for c in CONDITIONS]
            tasks += [dict(sample_id=r['sample_id'],kind='ta',condition=c) for c in ['joint','rr_off','pr_off']]
        number=len(ledger['submissions'])
        manifest=EXPANDED/f'tasks_{number:03}.json'
        write_json(manifest,tasks)
        if (EXPANDED/'STOP').exists():
            raise RuntimeError('STOP sentinel exists; do not queue more work')
        # Save intent before submitting to prevent duplicate work on an interrupted restart.
        entry=dict(manifest=str(manifest),samples=[r['sample_id'] for r in pending],task_count=len(tasks),status='submitting')
        ledger['submissions'].append(entry);write_json(ledger_path,ledger)
        routing=EXPANDED/'partition_routing.json'
        partition_args=[f"--partition={json.loads(routing.read_text())['discovered_partition']}"] if routing.exists() else []
        job=subprocess.check_output(['sbatch','--parsable','--priority=10000',*partition_args,f'--array=0-{len(tasks)-1}',
                     'claude_scripts/monitor_expanded_0924.sbatch',str(manifest)],text=True).strip().split(';')[0]
        entry.update(job_id=job,status='submitted');write_json(ledger_path,ledger)
        with (EXPANDED/f'monitor_{job}.log').open('a') as log:
            subprocess.Popen(['monitor_jobs',job],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('SUBMITTED',job,len(tasks),entry['samples'],flush=True)
        pending=[]

    submit_batch()
    with cf.ThreadPoolExecutor(max_workers=8) as executor:
        futures={executor.submit(group_prepare,group):group for group in groups.values()}
        for future in cf.as_completed(futures):
            group=futures[future]
            try:
                records=future.result()
            except Exception as exc:
                ledger['failures'].append(dict(samples=[r['sample_id'] for r in group],error_type=type(exc).__name__,error=str(exc)))
                write_json(ledger_path,ledger)
                print('PREPARATION FAILED',[r['sample_id'] for r in group],str(exc),flush=True)
                continue
            for r in records:
                write_json(EXPANDED/'inputs'/f"{r['sample_id']}.json",r)
                prepared.append(r);pending.append(r);ledger['prepared'].append(r['sample_id'])
                print('FROZEN',r['sample_id'],len(r['input_ids']),len(r['chunks']),flush=True)
            write_json(ledger_path,ledger)
            if len(pending)>=8:
                submit_batch()
    submit_batch()
    append_frozen(prepared)
    ledger['unresolved_samples']=[r['sample_id'] for r in data if not (EXPANDED/'inputs'/f"{r['sample_id']}.json").exists()]
    ledger['status']='submitted_with_preparation_failures' if ledger['unresolved_samples'] else 'all_released_records_submitted'
    write_json(ledger_path,ledger)
    print('COMPLETE',json.dumps(ledger),flush=True)


if __name__=='__main__':
    main()
