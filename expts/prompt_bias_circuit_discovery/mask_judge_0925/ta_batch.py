"""Thought Anchors scores for many traces with one model load.

Same scoring as monitor_pilot_0924.run.thought_anchors: suppress one source chunk's keys for every
later query under the matched background, and record the mean full-vocabulary KL per reader chunk.
"""
import argparse,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from ..monitor_pilot_0924 import run as original
from ..monitor_pilot_0924.core import CellPool,AttentionMask
from ..monitor_pilot_0924.prepare import MODEL,REVISION,digest

NEW=Path('results/prompt_bias_v2/mask_judge_0925')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--background',default='rr_off');ap.add_argument('--inputs',default=str(NEW/'inputs'));a=ap.parse_args()
 shard=int(os.environ.get('SLURM_ARRAY_TASK_ID',0))
 ids=json.loads(Path(a.manifest).read_text())[shard]
 model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager')
 model.config.use_cache=False;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 controller=AttentionMask(model)
 for sid in ids:
  d=NEW/'ta'/sid/a.background
  if (d/'complete.json').exists():continue
  d.mkdir(parents=True,exist_ok=True);begun=time.monotonic()
  r=json.loads((Path(a.inputs)/f'{sid}.json').read_text());checked=dict(r);sha=checked.pop('frozen_sha256');assert digest(checked)==sha
  x=torch.tensor([r['input_ids']],device='cuda')
  with torch.no_grad():
   controller.bias=None;ref=model.model(x,use_cache=False).last_hidden_state
   pool=CellPool(r,'joint','cuda');controller.bias=pool.additive(torch.ones(pool.count,device='cuda'));same=model.model(x,use_cache=False).last_hidden_state
   assert torch.equal(ref,same),'Native attention identity failed'
   del ref,same;controller.bias=None
   original.thought_anchors(model,controller,r,a.background,d,0)
  original.write_json(d/'complete.json',dict(sample_id=sid,background=a.background,frozen_sha256=sha,model=MODEL,revision=REVISION,job_id=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),status='complete',elapsed_seconds=time.monotonic()-begun))
  print('DONE',sid,round(time.monotonic()-begun,1),flush=True)

if __name__=='__main__':main()
