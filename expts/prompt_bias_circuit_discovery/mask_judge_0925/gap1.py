"""Sentence gap 1 for the monitor masks: reads within a reasoning chunk are always on, not learnable.

The stopped study's CellPool made within-chunk reads (cell reader == source) learnable, which is the
paper's sentence gap 0. Gap1Pool moves exactly those token reads from the eligible pool into the fixed-on
set and re-indexes the remaining between-chunk cells. Everything else (background, token self-reads,
first three keys, retention of 20% of eligible cells) is unchanged.
"""
import math
import torch
from ..monitor_pilot_0924.core import CellPool

class Gap1Pool(CellPool):
 retention=.2
 def __init__(self,record,condition,device='cpu',with_probe=False):
  super().__init__(record,condition,device,with_probe)
  size=self.length+(6 if with_probe else 0);n=len(self.chunks)
  tc=torch.full((size,),-1,device=device,dtype=torch.long)
  for i,c in enumerate(self.chunks):tc[c['start']:c['end']+1]=i
  same=(tc[:,None]==tc[None,:])&(tc[:,None]>=0)
  within=self.eligible&same
  self.eligible=self.eligible&~same
  self.fixed_on=self.fixed_on|within
  pairs=tc[:,None]*n+tc[None,:]
  codes=pairs[self.eligible].unique(sorted=True)
  self.pairs=[(int(c)//n,int(c)%n) for c in codes.cpu()]
  self.slots=torch.full((size,size),-1,device=device,dtype=torch.long)
  self.slots[self.eligible]=torch.searchsorted(codes,pairs[self.eligible])
  self.count=len(self.pairs);self.keep=math.floor(self.retention*self.count+.5)
  assert self.count>0 and self.fixed_on.diagonal().all() and all(q!=k for q,k in self.pairs)
