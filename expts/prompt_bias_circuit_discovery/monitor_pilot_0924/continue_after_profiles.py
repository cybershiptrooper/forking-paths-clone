"""Advance the authorized pilot only after all fixed profiles pass the compute check.

No scientific score/enrichment threshold selects runs. This launches fitting and
complete TA only; it never launches the expanded study or changes the protocol.
"""
import json
import subprocess
from pathlib import Path
from .prepare import ROOT, CONDITIONS

PROFILE_JOBS = ['88269', '88270', '88271']


def main():
    path = ROOT / 'continuation_ledger.json'
    if path.exists():
        raise RuntimeError('A continuation ledger already exists; refusing duplicate submission')
    status = subprocess.run(['monitor_jobs', *PROFILE_JOBS], check=False)
    if status.returncode:
        path.write_text(json.dumps({'status':'profile_job_failed', 'jobs':PROFILE_JOBS}, indent=2))
        raise RuntimeError('At least one profile failed; no continuation submitted')
    forecast = []
    for sid in ['D01', 'D02', 'D03', 'D04']:
        directory = ROOT / sid
        metadata = json.loads((directory / 'metadata.json').read_text())
        assert metadata['all_one_hidden_max_abs_error'] == 0
        fit_seconds = 0.
        for condition in CONDITIONS:
            data = json.loads((directory / f'{condition}_mask.json').read_text())
            assert data['steps'] == 50 and not data['final']
            fit_seconds += 950 * data['mean_step_seconds']
        ta_seconds = 0.
        for background in ['joint', 'rr_off', 'pr_off']:
            data = json.loads((directory / f'ta_{background}.json').read_text())
            samples = list(data['seconds_by_source'].values())
            assert samples
            ta_seconds += (data['total_sources'] - len(samples)) * sum(samples)/len(samples)
        # A fixed 50% runtime margin plus 10 minutes for load/evaluation, not a score gate.
        estimated_hours = (1.5 * (fit_seconds + ta_seconds) + 600) / 3600
        forecast.append({'sample':sid, 'remaining_fit_seconds':fit_seconds,
                         'remaining_ta_seconds':ta_seconds, 'guarded_gpu_hours':estimated_hours})
    raw = subprocess.check_output(['sacct', '-X', '-j', ','.join(PROFILE_JOBS),
               '--format=JobID%40,State,ElapsedRaw,AllocTRES%200', '-n', '-P'], text=True)
    wanted = {'88269', '88270_1', '88270_2', '88270_3', '88271'}
    rows = {}
    for line in raw.splitlines():
        parts = line.split('|')
        if parts[0] in wanted:
            rows[parts[0]] = parts
    assert set(rows) == wanted, 'Missing allocation records; do not guess consumed GPU time'
    assert all(parts[1] == 'COMPLETED' for parts in rows.values())
    consumed = sum(int(parts[2])/3600 for parts in rows.values())  # each profile requested exactly one GPU
    ledger = {'status':'forecast_checked', 'profile_gpu_hours':consumed, 'forecast':forecast,
              'continuation_max_gpu_hours':12, 'pilot_gpu_hour_cap':24,
              'scope':'D01-D04, all five masks to 1000 steps and complete background-matched TA',
              'judge_calls_submitted':False, 'expanded_study_submitted':False}
    if any(r['guarded_gpu_hours'] > 3 for r in forecast) or consumed + 12 > 24:
        ledger['status'] = 'compute_check_exceeded_no_submission'
        path.write_text(json.dumps(ledger, indent=2))
        print(json.dumps(ledger), flush=True)
        return
    # Persist an intention before the scheduler call: a restart cannot duplicate the batch.
    ledger['status'] = 'submitting'
    path.write_text(json.dumps(ledger, indent=2))
    job = subprocess.check_output(['sbatch', '--parsable', '--priority=10000',
                '--array=0-3%4', '--time=03:00:00', '--job-name=monitor-pilot-full',
                'slurm_scripts/monitor_pilot_0924_full.sbatch'], text=True).strip().split(';')[0]
    ledger.update(status='submitted', job_id=job)
    path.write_text(json.dumps(ledger, indent=2))
    print(json.dumps(ledger), flush=True)
    result = subprocess.run(['monitor_jobs', job], check=False)
    ledger['status'] = 'fits_and_TA_complete' if result.returncode == 0 else 'continuation_job_failed'
    path.write_text(json.dumps(ledger, indent=2))


if __name__ == '__main__':
    main()
