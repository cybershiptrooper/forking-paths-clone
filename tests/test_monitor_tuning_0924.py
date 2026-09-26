import math
import torch
from expts.prompt_bias_circuit_discovery.monitor_tuning_0924.worker import TuningPool,configure
from expts.prompt_bias_circuit_discovery.monitor_pilot_0924 import run
from expts.prompt_bias_circuit_discovery.monitor_pilot_0924.core import CellPool

def test_retention_changes_budget_without_changing_attention_scope():
 record=dict(input_ids=list(range(14)),prompt_length=7,chunks=[dict(id='P00',start=0,end=2),dict(id='P01',start=3,end=6),dict(id='R01',start=7,end=9),dict(id='R02',start=10,end=13)])
 old_pool,old_loss=run.CellPool,run.size_loss
 try:
  for retention in [.2,.5]:
   configure(retention)
   for condition in ['rr_on','rr_off']:
    pool=TuningPool(record,condition);ref=CellPool(record,condition)
    assert pool.pairs==ref.pairs and torch.equal(pool.fixed_on,ref.fixed_on)
    assert pool.keep==math.floor(retention*pool.count+.5)
    assert int(pool.hard(torch.zeros(pool.count)).sum())==pool.keep
    assert int(pool.random(0).sum())==pool.keep
   alpha=torch.full((10,),math.log(retention/(1-retention))+(2/3)*math.log(.1/1.1),requires_grad=True)
   assert float(run.size_loss(alpha).detach())<1e-10
 finally:run.CellPool,run.size_loss=old_pool,old_loss

import unittest
import json
from pathlib import Path
import numpy as np
from expts.prompt_bias_circuit_discovery.monitor_tuning_0924.metrics import threshold,folds
from expts.prompt_bias_circuit_discovery.monitor_tuning_0924.confirm_metrics import cluster_draw

class TuningTests(unittest.TestCase):
 def test_retention_scope(self):test_retention_changes_budget_without_changing_attention_scope()
 def test_threshold_calibration_does_not_assume_half(self):
  self.assertGreater(threshold([0,0,1,1],[.7,.8,.95,.99]),.8)
  self.assertLessEqual(threshold([0,0,1,1],[.7,.8,.95,.99]),.95)
 def test_cluster_bootstrap_keeps_paired_variants(self):
  rows=[dict(base_id='a',Y=0),dict(base_id='a',Y=1),dict(base_id='b',Y=0),dict(base_id='c',Y=1)]
  rng=np.random.default_rng(123)
  for _ in range(100):
   ix=cluster_draw(rows,rng);self.assertEqual(int((ix==0).sum()),int((ix==1).sum()))
 def test_folds_hold_out_every_stratum_and_question(self):
  rows=[dict(task=task,Y=y,hint_letter=letter,base_id=f'{task}-{y}-{letter}-{i}',sample_id=f'{task}-{y}-{letter}-{i}') for task in [4,5] for y in [0,1] for letter in ['A','B'] for i in range(6)]
  assigned=folds(rows)
  for task in [4,5]:
   for f in range(6):
    held=[r for r in rows if r['task']==task and assigned[r['sample_id']]==f]
    self.assertEqual(len(held),4);self.assertEqual({(r['Y'],r['hint_letter']) for r in held},{(0,'A'),(0,'B'),(1,'A'),(1,'B')})
