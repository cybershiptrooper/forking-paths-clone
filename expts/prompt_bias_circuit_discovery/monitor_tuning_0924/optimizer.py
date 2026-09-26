"""Predeclared fixed-penalty SNP diagnostic; no Lagrangian or judge-loss training."""
import json,math,time
import torch
from ..monitor_pilot_0924 import run
from ..monitor_pilot_0924.core import sample_gates,trace_nll

# The worker sets this process-local configuration before a single fit.
CONFIG={}
def fit(model,controller,record,condition,steps,directory):
 pool=run.CellPool(record,condition,'cuda');retention=CONFIG['retention'];peak=CONFIG['penalty_peak'];sparse_init=CONFIG['sparse_init']
 torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 initial=math.log(retention/(1-retention))+(2/3)*math.log(.1/1.1) if sparse_init else 2.
 alpha=torch.full((pool.count,),initial,device='cuda',requires_grad=True)
 opt=torch.optim.Adam([alpha],lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 checkpoint=directory/f'{condition}.pt';logpath=directory/f'{condition}_training.json';history=[];start=0
 if checkpoint.exists():
  state=torch.load(checkpoint,weights_only=False,map_location='cpu');assert state['frozen_sha256']==record['frozen_sha256'] and state['condition']==condition
  with torch.no_grad():alpha.copy_(state['alpha'].to('cuda'))
  opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);start=state['step'];history=json.loads(logpath.read_text())[:start]
 ids=torch.tensor([record['input_ids']],device='cuda');model.train()
 for step in range(start,steps):
  run.sync();begun=time.monotonic();opt.zero_grad(set_to_none=True);losses=[]
  for _ in range(4):
   gates=sample_gates(alpha);controller.bias=pool.additive(gates);loss=trace_nll(model,ids,record['prompt_length']);(loss/4).backward();losses.append(float(loss.detach()))
  assert torch.isfinite(alpha.grad).all(),'Nonfinite task gradient';norm=float(alpha.grad.norm())
  if step==0:assert norm>0,'Trace loss has no gradient'
  coef=peak if sparse_init else peak*run.size_coefficient(step)/1000.
  size_grad,=torch.autograd.grad(coef*run.size_loss(alpha),alpha);opt.step()
  with torch.no_grad():alpha.sub_(.1*size_grad)
  assert torch.isfinite(alpha).all(),'Nonfinite gate parameters';controller.bias=None;run.sync()
  history.append(dict(step=step+1,nll=sum(losses)/4,expected_active=float(torch.sigmoid(alpha.detach()-(2/3)*math.log(.1/1.1)).sum()),task_gradient_norm=norm,size_coefficient=coef,seconds=time.monotonic()-begun,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
  if (step+1)%5==0 or step+1==steps:
   run.write_json(logpath,history);run.checkpoint_save(checkpoint,alpha,opt,step+1,record,condition);print(record['sample_id'],condition,json.dumps(history[-1]),flush=True)
 model.eval();binary=pool.hard(alpha)
 result=dict(steps=steps,eligible_cells=pool.count,kept_cells=int(binary.sum()),pairs=[[record['chunks'][q]['id'],record['chunks'][k]['id']] for q,k in pool.pairs],binary=binary.cpu().tolist(),alpha=alpha.detach().cpu().tolist(),diagnostics=run.diagnostics(model,controller,record,pool,binary),final=steps==1000,optimizer_config=dict(CONFIG),total_training_seconds=sum(r['seconds'] for r in history),mean_step_seconds=sum(r['seconds'] for r in history)/len(history))
 run.write_json(directory/f'{condition}_mask.json',result);return result
