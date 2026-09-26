"""Isolated fixed-sparsity worker. Does not modify the running original sweep."""
import argparse,json,os,math,time,hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from ..monitor_pilot_0924 import run as original
from ..monitor_pilot_0924.core import CellPool,AttentionMask
from ..monitor_pilot_0924.prepare import MODEL,REVISION,digest
from .cohort import ROOT

class TuningPool(CellPool):
 retention=.2
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.keep=math.floor(self.retention*self.count+.5)

def configure(retention):
 assert retention in [.2,.5]
 TuningPool.retention=retention
 original.CellPool=TuningPool
 def size_loss(alpha):
  expected=torch.sigmoid(alpha-(2/3)*math.log(.1/1.1)).sum()
  return (expected-retention*alpha.numel()).square()/alpha.numel()
 original.size_loss=size_loss

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--index',type=int);a=ap.parse_args()
 task=json.loads(Path(a.manifest).read_text())[a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])]
 r=json.loads((ROOT/'inputs'/f"{task['sample_id']}.json").read_text()); checked=dict(r);sha=checked.pop('frozen_sha256');assert digest(checked)==sha
 d=ROOT/task['kind']/r['sample_id']/task['recipe'];d.mkdir(parents=True,exist_ok=True)
 if (d/'complete.json').exists(): return
 assert not (ROOT/'STOP').exists()
 configure(task.get('retention',.2));begun=time.monotonic()
 meta=dict(task=task,frozen_sha256=sha,model=MODEL,revision=REVISION,job_id=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),implementation_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in [Path(__file__),Path(original.__file__),Path(__file__).parent.parent/'monitor_pilot_0924/core.py'])).hexdigest())
 original.write_json(d/'metadata.json',meta)
 try:
  assert 'H100' in meta['gpu']
  model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager')
  model.config.use_cache=False;model.eval();assert model.config.attention_dropout==0
  for p in model.parameters():p.requires_grad_(False)
  ids=torch.tensor([r['input_ids']],device='cuda')
  with torch.no_grad():ref=model.model(ids,use_cache=False).last_hidden_state
  controller=AttentionMask(model)
  with torch.no_grad():
   pool=TuningPool(r,'joint','cuda');controller.bias=pool.additive(torch.ones(pool.count,device='cuda'));actual=model.model(ids,use_cache=False).last_hidden_state
   assert torch.equal(ref,actual),'Native attention identity failed'
  del ref,actual;controller.bias=None;meta['all_one_hidden_max_abs_error']=0.
  if task['kind']=='fits':
   pool=TuningPool(r,task['condition'],'cuda')
   controls={'clean':original.diagnostics(model,controller,r,None,None)}
   for name,gates in [('full',torch.ones(pool.count,device='cuda')),('empty',torch.zeros(pool.count,device='cuda'))]+[(f'random{i}',pool.random(i)) for i in range(5)]:
    controls[name]=original.diagnostics(model,controller,r,pool,gates)
   original.write_json(d/'controls.json',controls)
   model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
   step=torch.optim.Adam.step
   def guarded(opt,*args,**kwargs):
    if (ROOT/'STOP').exists():raise RuntimeError('Tuning STOP sentinel exists')
    return step(opt,*args,**kwargs)
   torch.optim.Adam.step=guarded
   original.fit(model,controller,r,task['condition'],1000,d)
  else:original.thought_anchors(model,controller,r,task['condition'],d,0)
  meta.update(status='complete',elapsed_seconds=time.monotonic()-begun);original.write_json(d/'complete.json',meta)
 except Exception as exc:
  meta.update(status='failed',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-begun);original.write_json(d/'failure.json',meta)
  if isinstance(exc,AssertionError) or 'Nonfinite' in str(exc):(ROOT/'STOP').write_text(json.dumps(meta,indent=2))
  raise
if __name__=='__main__':main()
