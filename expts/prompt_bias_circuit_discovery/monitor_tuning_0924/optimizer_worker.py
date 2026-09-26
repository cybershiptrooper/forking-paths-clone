"""Run a separately versioned optimizer variant through the validated worker."""
import argparse,json,os
from pathlib import Path
from . import worker,optimizer

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--index',type=int);a=ap.parse_args()
 task=json.loads(Path(a.manifest).read_text())[a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])]
 optimizer.CONFIG=dict(retention=task['retention'],penalty_peak=task['penalty_peak'],sparse_init=task['sparse_init'])
 from .cohort import ROOT,save
 import hashlib
 save(ROOT/task['kind']/task['sample_id']/task['recipe']/'optimizer_source.json',dict(source=Path(optimizer.__file__).read_text(),sha256=hashlib.sha256(Path(optimizer.__file__).read_bytes()).hexdigest()))
 worker.original.fit=optimizer.fit
 worker.main()
if __name__=='__main__':main()
